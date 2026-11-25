from pathlib import Path
import sys
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parents[2]
load(name='inkling7cd_ops',sources=[str(root/'src/models/inkling2/native_executor_stage7cd.cpp')],extra_cflags=['-O3'],with_cuda=False,is_python_module=False,build_directory=sys.argv[1],verbose=True)
