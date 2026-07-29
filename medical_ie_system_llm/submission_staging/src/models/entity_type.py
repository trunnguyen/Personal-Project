from enum import Enum

class EntityType(Enum):
    SYMPTOM = "TRIỆU_CHỨNG"

    DIAGNOSIS = "CHẨN_ĐOÁN"

    DRUG="THUỐC"

    LAB_TEST="TÊN_XÉT_NGHIỆM"

    LAB_RESULT="KẾT_QUẢ_XÉT_NGHIỆM"
