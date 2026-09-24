from frontiermem.external_data import normalize_rpeval_native_label


def test_rpeval_chinese_supportive_variants():
    assert normalize_rpeval_native_label("支持性偏好") == "SUPPORT"
    assert normalize_rpeval_native_label("支支持持性性偏偏好好") == "SUPPORT"
