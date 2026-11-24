"""Stage5F/5G contract tests; do not depend on CUDA or full 20GB checkpoint."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

from ninfer_artifact_bridge import swizzle_nvfp4_128_rows
from real_carrier_stage5f import (
    Carrier5F, HIDDEN, INTERMEDIATE, KIND_MXFP8, KIND_NVFP4,
    bf16_to_fp32, fp32_values, f32_bytes, router_scores,
)
from prepare_real_moe_stage5f import (
    MAGIC, HEADER, expected_sizes, verify_bundle,
)


class Obj:
    def __init__(self, id, data: bytes, *, kind="resource", fmt="", layout="", shape=()):
        self.id=id
        self.offset=0
        self.bytes=len(data)
        self.kind=kind
        self.format=fmt
        self.layout=layout
        self.shape=shape
        self.data=data


class FakeArtifact:
    def __init__(self, storage_contract, objects, banks=()):
        self._objects={o.id:o for o in objects}
        at=0
        for o in objects:
            o.offset=at
            at+=o.bytes+16
        self.directory=type("Directory", (), {
            "metadata":{"storage_contract":storage_contract,"inkling2_banks":list(banks)}
        })()

    def object(self, id):
        return self._objects[id]

    def read_range(self, offset, size):
        for o in self._objects.values():
            if o.offset <= offset < o.offset + o.bytes and offset + size <= o.offset+o.bytes:
                return o.data[offset-o.offset:offset-o.offset+size]
        raise ValueError("read outside fake object")

    def read_object(self,id):
        return self._objects[id].data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def banks_fixture(nv: bool, projection: str):
    n,k=HIDDEN,HIDDEN if projection=="w13" else INTERMEDIATE
    layer=0 if nv else 20
    prefix=f"inkling2/layer{layer:02d}/{projection}"
    fmt=KIND_NVFP4 if nv else KIND_MXFP8
    count=2
    if nv:
        one_code=n*k//2
        one_scale=n*k//16
        codes=bytes((0x21 if e==0 else 0x84) for e in range(count) for _ in range(one_code))
        scales_natural=[bytes((0x30 + (r + g + e)%7) for r in range(n) for g in range(k//16))
                        for e in range(count)]
        scales_swizzled=b"".join(
            swizzle_nvfp4_128_rows(scales[i:i+128*k//16],k)
            for scales in scales_natural for i in range(0,len(scales),128*k//16))
        scalars=struct.pack("<ffff",2.,8.,3.,8.)
        objects=[Obj(prefix+"/nvfp4",codes+scales_swizzled+struct.pack("<ff",4.,8.),
                     kind="tensor",fmt="nvfp4",layout="batched_block_scale_k16_m128x4_v1",
                     shape=(count,n,k)),
                 Obj(prefix+"/original_amax_denominator_fp32",scalars)]
        expected=(codes[:one_code],scales_natural[0],2.,8.)
        record={"layer":layer,"projection":projection,"expert_ids":[0,1],
                "shape":[count,n,k],"quantization":"nvfp4_e2m1_g16_e4m3"}
    else:
        one_code=n*k
        one_scale=n*k//32
        codes=b"\x38"*one_code+b"\xb8"*one_code
        scales=b"\x7f"*one_scale+b"\x80"*one_scale
        objects=[Obj(prefix+"/mxfp8_codes",codes),
                 Obj(prefix+"/mxfp8_ue8m0_scales",scales)]
        expected=(codes[:one_code],scales[:one_scale],1.,1.)
        record={"layer":layer,"projection":projection,"expert_ids":[0,1],
                "shape":[count,n,k],"quantization":"mxfp8_e4m3_g32_ue8m0",
                "scale_group":32}
    return objects,record,expected


def fake_pair(objects, records):
    non=FakeArtifact("inkling2_hf_nonexpert_v3_v1",[])
    experts=FakeArtifact("inkling2_ninfer_v3_expert_carrier_v1",objects,records)
    return Carrier5F(non,experts)


class TestRealWeightsStage5F(unittest.TestCase):
    def test_exact_bf16_word_conversion(self):
        words=struct.pack("<5H",0x3f80,0xbf80,0x0001,0x7f80,0x8000)
        converted=bf16_to_fp32(words)
        self.assertEqual(converted,struct.pack("<5I",0x3f800000,0xbf800000,
                                               0x00010000,0x7f800000,0x80000000))
        with self.assertRaisesRegex(ValueError,"even length"):
            bf16_to_fp32(b"x")

    def test_fp32_roundtrip_little_endian(self):
        values=(1.,-0.5,0.,127.25)
        self.assertEqual(fp32_values(f32_bytes(values)),values)

    def test_router_bias_selects_but_does_not_change_score(self):
        x=(0.,)*HIDDEN
        router=b"\x00"*(128*HIDDEN*4)
        b=[0.]*128
        b[91]=1.
        ranked=router_scores(x,router,f32_bytes(b))
        self.assertEqual(ranked[0][2],91)
        self.assertEqual(ranked[0][1],.5)
        self.assertEqual([v[2] for v in ranked[:3]],[91,0,1])
        with self.assertRaises(ValueError):
            router_scores(x,router,b"\x00")

    def test_nvfp4_native_v3_swizzle_restore_and_original_scalars(self):
        objs,rec,(expected_codes,expected_scales,amax,den)=banks_fixture(True,"w2")
        view=fake_pair(objs,[rec])
        a,b=view.bank(0,"w2",[0,1])
        self.assertEqual((a.codes,a.scales,a.amax,a.denominator),
                         (expected_codes,expected_scales,amax,den))
        self.assertEqual((b.amax,b.denominator),(3.,8.))
        self.assertNotEqual(a.codes,b.codes)
        with self.assertRaisesRegex(ValueError,"requested expert missing"):
            view.bank(0,"w2",[2])

    def test_mxfp8_raw_planes_and_expert_indexing(self):
        objs,rec,(codes,scales,_,_)=banks_fixture(False,"w2")
        view=fake_pair(objs,[rec])
        first,second=view.bank(20,"w2",[0,1])
        self.assertEqual(first.codes,codes)
        self.assertEqual(first.scales,scales)
        self.assertTrue(all(b==0xb8 for b in second.codes[:100]))
        self.assertTrue(all(b==0x80 for b in second.scales[:100]))
        with self.assertRaisesRegex(ValueError,"repeated"):
            view.bank(20,"w2",[0,0])

    def test_bank_dtype_mismatch_fail_closed(self):
        objs,rec,_=banks_fixture(True,"w2")
        objs[0].layout="wrong_layout"
        with self.assertRaisesRegex(ValueError,"layout mismatch"):
            fake_pair(objs,[rec]).bank(0,"w2",[0])
        objs,rec,_=banks_fixture(False,"w2")
        rec["scale_group"]=16
        with self.assertRaisesRegex(ValueError,"contract"):
            fake_pair(objs,[rec]).bank(20,"w2",[0])

    def test_rejects_invalid_carrier_contract(self):
        o=FakeArtifact("wrong",[])
        e=FakeArtifact("inkling2_ninfer_v3_expert_carrier_v1",[])
        with self.assertRaisesRegex(ValueError,"non-expert"):
            Carrier5F(o,e)

    def test_full_geometry_stage5e_contract(self):
        for fmt in (KIND_NVFP4,KIND_MXFP8):
            sizes=expected_sizes(fmt,32)
            self.assertEqual(sizes["header.bin"],36)
            self.assertEqual(sizes["router.f32"],128*2304*4)
            self.assertEqual(sizes["w13_codes.u8"],32*2304*2304//(2 if fmt==1 else 1))
            self.assertEqual(sizes["w2_scales.u8"],32*2304*1152//(16 if fmt==1 else 32))
        with self.assertRaises(ValueError):
            expected_sizes(3,32)
        with self.assertRaises(ValueError):
            expected_sizes(1,7)

    def test_full_prepare_bundle_with_fake_real_carriers(self):
        # Exercises the complete executable Stage-5F Python path (including
        # streamed v3 range reads and SHA256), without a 20GB checkpoint.
        from prepare_real_moe_stage5f import create_bundle
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"config.json").write_text(json.dumps({"text_config":{"route_scale":8.0}}))
            embedded=struct.pack("<H",0x3f80)*HIDDEN
            router=b"\x00"*(128*HIDDEN*2)
            bias=f32_bytes([8. if e<8 else 0. for e in range(128)])
            nobj=[
                Obj("hf/model.llm.embed.weight", embedded,kind="tensor",fmt="bf16",shape=(1,HIDDEN)),
                Obj("hf/model.llm.layers.0.mlp.gate.weight",router,kind="tensor",fmt="bf16",shape=(128,HIDDEN)),
                Obj("hf/model.llm.layers.0.mlp.gate.bias",bias,kind="tensor",fmt="fp32",shape=(128,)),
            ]
            non=FakeArtifact("inkling2_hf_nonexpert_v3_v1",nobj)
            objects=[];records=[]
            for proj in ("w13","w2"):
                k=HIDDEN if proj=="w13" else INTERMEDIATE
                prefix=f"inkling2/layer00/{proj}"
                count=8
                stride_code=HIDDEN*k//2
                scales_row=bytes([0x38])*(HIDDEN*k//16)
                one_swizzled=b"".join(swizzle_nvfp4_128_rows(scales_row[i:i+128*k//16],k)
                                       for i in range(0,len(scales_row),128*k//16))
                codes=b"\x21"*(count*stride_code)
                scales=one_swizzled*count
                objects += [Obj(prefix+"/nvfp4",codes+scales+struct.pack("<8f",*([4.]*8)),
                                kind="tensor",fmt="nvfp4",layout="batched_block_scale_k16_m128x4_v1",
                                shape=(count,HIDDEN,k)),
                            Obj(prefix+"/original_amax_denominator_fp32",struct.pack("<16f",*([1.,4.]*8)))]
                records.append({"layer":0,"projection":proj,"expert_ids":list(range(8)),
                                "shape":[count,HIDDEN,k],"quantization":"nvfp4_e2m1_g16_e4m3"})
            ex=FakeArtifact("inkling2_ninfer_v3_expert_carrier_v1",objects,records)
            non.artifact_id=b"A"*16
            ex.artifact_id=b"B"*16
            reader_module=types.ModuleType("tools.artifact.reader")
            reader_module.Artifact=lambda path: non if str(path)=="non.ninfer" else ex
            # Real class is imported lazily; a deliberately tiny fake NInfer
            # reader is sufficient to validate Stage5F's byte-range contract.
            fake_tools=types.ModuleType("tools");fake_tools.__path__=[]
            fake_art=types.ModuleType("tools.artifact");fake_art.__path__=[]
            with patch.dict("sys.modules",{
                "tools":fake_tools,"tools.artifact":fake_art,"tools.artifact.reader":reader_module,
            }),patch("real_carrier_stage5f.validate_complete_carriers",return_value={
                "roles":["dummy"],
                "checkpoint":{"model_index_sha256":"hash"},
                "coverage":{"metadata_complete":True,"bank_manifest_matches":True},
            }):
                output=root/"bundle"
                meta=create_bundle(checkpoint_root=root,nonexpert_path=Path("non.ninfer"),
                                   experts_path=Path("experts.ninfer"),banks_path=root,
                                   output_path=output,layer=0,token_id=0,count=8)
                self.assertEqual(meta["resident_ids"],list(range(8)))
                self.assertEqual(verify_bundle(output)["layer"],0)
                self.assertEqual((output/"hidden.f32").read_bytes(),f32_bytes([1.]*HIDDEN))
                self.assertEqual((output/"w13_codes.u8").stat().st_size,8*HIDDEN*HIDDEN//2)
                with self.assertRaises(FileExistsError):
                    create_bundle(checkpoint_root=root,nonexpert_path=Path("non.ninfer"),
                                  experts_path=Path("experts.ninfer"),banks_path=root,
                                  output_path=output,layer=0,token_id=0,count=8)

    def test_manifest_detects_corruption_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            fmt=1;count=8
            size=expected_sizes(fmt,count)
            payloads={name:b"\x00"*n for name,n in size.items()}
            payloads["header.bin"]=HEADER.pack(MAGIC,1,0,fmt,count,HIDDEN,INTERMEDIATE,10)
            mapping=[-1]*128
            for e in range(count):mapping[e]=e
            payloads["resident_map.i32"]=struct.pack("<128i",*mapping)
            payloads["predicted_ids.i32"]=struct.pack("<8i",*range(8))
            files={}
            for name,data in payloads.items():
                (root/name).write_bytes(data)
                files[name]={"bytes":len(data),"sha256":sha256(data).hexdigest()}
            meta={"contract":"inkling2_real_moe_stage5fg_v1", "files":files,
                  "resident_ids":list(range(count)),"layer":0}
            (root/"manifest.json").write_text(json.dumps(meta))
            self.assertEqual(verify_bundle(root)["layer"],0)
            with (root/"hidden.f32").open("r+b") as f:
                f.write(b"not")
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):
                verify_bundle(root)
            meta["files"].pop("hidden.f32")
            (root/"manifest.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,"missing or additional"):
                verify_bundle(root)


if __name__ == "__main__":
    unittest.main()
