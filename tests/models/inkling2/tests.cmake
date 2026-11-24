ninfer_add_test(ninfer_inkling2_contract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_contract.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_loading ninfer::json)
set_tests_properties(ninfer_inkling2_contract_test PROPERTIES LABELS "model;cpu;inkling2")
