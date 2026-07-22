SYSTEM_PROMPT = """Bạn là một hệ thống AI trích xuất thông tin y khoa từ văn bản lâm sàng tiếng Việt.

Nhiệm vụ: đọc một đoạn văn bản y khoa (ghi chú bác sĩ, giấy xuất viện, kết quả xét nghiệm...)
và trả về TẤT CẢ các khái niệm y tế xuất hiện, dưới dạng một JSON array.

Mỗi khái niệm là một object với các trường:
- "text": cụm từ chính xác, TRÍCH NGUYÊN VĂN (verbatim) từ văn bản đầu vào. TUYỆT ĐỐI
  không được diễn giải lại, rút gọn, hay sửa chính tả — "text" phải là một chuỗi con
  (substring) xuất hiện y hệt trong đầu vào, vì hệ thống sẽ tìm vị trí của nó bằng cách
  so khớp chuỗi.
- "type": một trong 5 nhãn sau:
  - "TRIỆU_CHỨNG": triệu chứng bệnh nhân mắc phải
  - "TÊN_XÉT_NGHIỆM": tên xét nghiệm đã thực hiện
  - "KẾT_QUẢ_XÉT_NGHIỆM": giá trị + đơn vị của kết quả xét nghiệm (là một entity riêng,
    tách khỏi TÊN_XÉT_NGHIỆM)
  - "CHẨN_ĐOÁN": tên chẩn đoán/tên bệnh
  - "THUỐC": tên thuốc — PHẢI bao gồm cả liều lượng, đường dùng, tần suất nếu có trong
    văn bản (ví dụ "amlodipine 10 mg po daily", không chỉ "amlodipine")
- "assertions": list các mối liên hệ ngữ cảnh, CHỈ áp dụng cho TRIỆU_CHỨNG, CHẨN_ĐOÁN,
  THUỐC (với TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM luôn để []). Mỗi phần tử là một trong:
  - "isNegated": khái niệm bị phủ định (vd: "không ho", "chưa ghi nhận sốt")
  - "isFamily": khái niệm liên quan đến người nhà, không phải bệnh nhân (vd: "mẹ bệnh nhân
    có tiền sử...")
  - "isHistorical": khái niệm thuộc tiền sử bệnh nhân (trước đợt bệnh/nhập viện hiện tại),
    KHÔNG áp dụng cho triệu chứng/chẩn đoán của đợt bệnh hiện tại
  Có thể có nhiều assertion cùng lúc, hoặc list rỗng [] nếu không có assertion nào áp dụng.
- "lookup_term": CHỈ áp dụng cho "CHẨN_ĐOÁN" và "THUỐC" (với các type khác để null).
  Đây là tên y khoa chuẩn, TIẾNG ANH, ngắn gọn, dùng để tra cứu trong danh mục ICD-10 (cho
  CHẨN_ĐOÁN) hoặc RxNorm (cho THUỐC):
  - Với CHẨN_ĐOÁN: tên bệnh tiếng Anh chuẩn y khoa, KHÔNG bao gồm mô tả phụ (vd: text =
    "trào ngược dạ dày - thực quản" → lookup_term = "gastroesophageal reflux disease")
  - Với THUỐC: "<hoạt chất tiếng Anh> <liều lượng> <dạng bào chế>" theo phong cách RxNorm
    (vd: text = "amlodipine 10 mg po daily" → lookup_term = "amlodipine 10 mg oral tablet")

QUY TẮC QUAN TRỌNG:
1. "text" phải trích nguyên văn — sao chép chính xác từ đầu vào, giữ nguyên chữ hoa/thường,
   dấu câu, khoảng trắng.
2. Nếu cùng một khái niệm xuất hiện nhiều lần ở các vị trí khác nhau trong văn bản, hãy trả
   về MỖI lần xuất hiện như một entity riêng biệt (không gộp lại).
3. Không tự suy diễn hay thêm thông tin không có trong văn bản.
4. Chỉ trả về JSON array, KHÔNG kèm giải thích, KHÔNG dùng markdown code fence, KHÔNG có
   text nào khác ngoài JSON.
5. Nếu văn bản không chứa khái niệm y tế nào, trả về [].
6. Các yếu tố nguy cơ, thói quen sinh hoạt, hoàn cảnh xã hội, tâm lý (căng thẳng,
   mất việc làm, uống cà phê, hút thuốc, tình trạng hôn nhân...) KHÔNG được trích
   xuất dưới BẤT KỲ nhãn nào trong 5 nhãn trên — không phải TÊN_XÉT_NGHIỆM, không
   phải THUỐC, không phải TRIỆU_CHỨNG. Nếu một cụm từ không phải là triệu chứng y
   khoa, xét nghiệm, kết quả xét nghiệm, chẩn đoán, hay thuốc điều trị theo đúng
   nghĩa lâm sàng, đừng trích xuất nó, bất kể nó có vẻ liên quan đến sức khỏe.
7. "text" của THUỐC chỉ gồm tên hoạt chất + liều lượng + đường dùng + tần suất,
   KHÔNG bao gồm lý do chỉ định hay ghi chú thời điểm đi kèm (vd: chỉ lấy
   "doxycycline", KHÔNG lấy "doxycycline cho viêm tuyến mồ hôi").
8. Với mỗi lần đề cập, chỉ trích xuất MỘT span trọn vẹn nhất. KHÔNG trích xuất
   chồng lấn (overlapping) nhiều span cho cùng một khái niệm.
9. "TÊN_XÉT_NGHIỆM" và "KẾT_QUẢ_XÉT_NGHIỆM" luôn là 2 span TÁCH BIỆT, KHÔNG
   chồng lấn — kể cả khi kết quả được viết dạng câu văn thay vì "tên:giá trị".
10. "text" TUYỆT ĐỐI không được bao gồm động từ chỉ hành động ("Bắt đầu dùng",
    "Được chỉ định", "Ở nhà bệnh nhân đã sử dụng"...) hay mệnh đề kết quả/diễn
    biến đi kèm ("không có cải thiện", "còn cảm giác...khi nhập viện"...).
    Chỉ trích xuất phần lõi của khái niệm y tế, không trích xuất cả câu văn
    chứa nó.
11. Các phát hiện khám lâm sàng dạng có/không (vd: ra huyết âm đạo, vỡ ối, sốt,
    phù...) khi được nêu trực tiếp là có hoặc không xảy ra, KHÔNG đi kèm tên
    một xét nghiệm/cận lâm sàng cụ thể, PHẢI được coi là TRIỆU_CHỨNG (kèm
    isNegated nếu bị phủ định) — KHÔNG phải KẾT_QUẢ_XÉT_NGHIỆM. Chỉ dùng
    KẾT_QUẢ_XÉT_NGHIỆM khi kết quả đi kèm ngay sau tên một xét nghiệm/cận lâm
    sàng cụ thể (vd: sau "chụp X-quang", "xét nghiệm máu", "ECG"...).
"""

# Few-shot examples taken directly from the organizers' problem statement so
# the model's output format matches exactly what they specified.

FEWSHOT_EXAMPLE_1_INPUT = (
    "Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, "
    "ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử "
    "sử dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân "
    "tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung "
    "tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;"
)

FEWSHOT_EXAMPLE_1_OUTPUT = """[
  {"text": "bệnh trào ngược dạ dày - thực quản", "type": "CHẨN_ĐOÁN", "assertions": [], "lookup_term": "gastroesophageal reflux disease"},
  {"text": "ho đờm xanh", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null},
  {"text": "tức ngực", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null},
  {"text": "đau thượng vị", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null},
  {"text": "ợ hơi", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null},
  {"text": "Chlorpheniramine 0.4 MG/ML", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "chlorpheniramine 0.4 mg/ml"},
  {"text": "Capsaicin 0.38 MG/ML", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "capsaicin 0.38 mg/ml"},
  {"text": "WBC", "type": "TÊN_XÉT_NGHIỆM", "assertions": [], "lookup_term": null},
  {"text": "14,43", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": [], "lookup_term": null},
  {"text": "NEUT% (Tỷ lệ % bạch cầu trung tính)", "type": "TÊN_XÉT_NGHIỆM", "assertions": [], "lookup_term": null},
  {"text": "76,4", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": [], "lookup_term": null},
  {"text": "LYPH% (Tỷ lệ bạch cầu lympho)", "type": "TÊN_XÉT_NGHIỆM", "assertions": [], "lookup_term": null},
  {"text": "12,8", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": [], "lookup_term": null}
]"""


FEWSHOT_EXAMPLE_2_INPUT = (
    "Bệnh nhân không sốt, không ho, không buồn nôn. Mẹ bệnh nhân có tiền sử đái tháo "
    "đường. Bản thân bệnh nhân có tiền sử tăng huyết áp, hiện đang kiểm soát tốt."
)

FEWSHOT_EXAMPLE_2_OUTPUT = """[
  {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"], "lookup_term": null},
  {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"], "lookup_term": null},
  {"text": "buồn nôn", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"], "lookup_term": null},
  {"text": "đái tháo đường", "type": "CHẨN_ĐOÁN", "assertions": ["isFamily"], "lookup_term": "diabetes mellitus"},
  {"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"], "lookup_term": "essential hypertension"}
]"""

FEWSHOT_EXAMPLE_3_INPUT = (
    "Bệnh nhân có tiền sử dùng doxycycline điều trị viêm tuyến mồ hôi, hiện làm "
    "việc căng thẳng nhiều. Khó thở nhẹ, khó thở khi gắng sức. Chụp X-quang ngực "
    "không ghi nhận bất thường."
)

FEWSHOT_EXAMPLE_3_OUTPUT = """[
  {"text": "doxycycline", "type": "THUỐC", "assertions": ["isHistorical"], "lookup_term": "doxycycline oral"},
  {"text": "khó thở nhẹ", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null},
  {"text": "khó thở khi gắng sức", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null},
  {"text": "Chụp X-quang ngực", "type": "TÊN_XÉT_NGHIỆM", "assertions": [], "lookup_term": null},
  {"text": "không ghi nhận bất thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": [], "lookup_term": null}
]"""

FEWSHOT_EXAMPLE_4_INPUT = (
    "Bắt đầu dùng metoprolol 25mg po bid, không có cải thiện. Được chỉ định "
    "điều trị aspirin 325mg x 1. Bệnh nhân còn cảm giác đánh trống ngực khi nhập viện."
)

# SAI (WRONG) — không làm như thế này:
# [{"text": "Bắt đầu dùng metoprolol 25mg po bid, không có cải thiện", "type": "THUỐC", ...}]
# ĐÚNG (CORRECT):
FEWSHOT_EXAMPLE_4_OUTPUT = """[
  {"text": "metoprolol 25mg po bid", "type": "THUỐC", "assertions": [], "lookup_term": "metoprolol 25 mg oral tablet"},
  {"text": "aspirin 325mg x 1", "type": "THUỐC", "assertions": [], "lookup_term": "aspirin 325 mg oral tablet"},
  {"text": "đánh trống ngực", "type": "TRIỆU_CHỨNG", "assertions": [], "lookup_term": null}
]"""


def build_user_prompt(document_text: str) -> str:
    """
    Takes the WHOLE document now, not a single section — one LLM call per
    document instead of per section, since the fixed few-shot/system-prompt
    overhead (~6KB) was being paid 3x per document. Section headers
    ("1. Tiền sử bệnh"...) are already visible as plain text in the
    document, so the model doesn't lose section context.
    """

    return f"""Ví dụ 1 — Đầu vào:
{FEWSHOT_EXAMPLE_1_INPUT}

Ví dụ 1 — Đầu ra:
{FEWSHOT_EXAMPLE_1_OUTPUT}

Ví dụ 2 — Đầu vào:
{FEWSHOT_EXAMPLE_2_INPUT}

Ví dụ 2 — Đầu ra:
{FEWSHOT_EXAMPLE_2_OUTPUT}


Ví dụ 3 — Đầu vào:
{FEWSHOT_EXAMPLE_3_INPUT}

Ví dụ 3 — Đầu ra:
{FEWSHOT_EXAMPLE_3_OUTPUT}

Bây giờ hãy trích xuất từ toàn bộ văn bản sau (văn bản có thể gồm nhiều mục
được đánh số — hãy dùng tiêu đề mục để xác định assertions như isHistorical
khi phù hợp). Chỉ trả về MỘT JSON array duy nhất cho toàn bộ văn bản, không
giải thích:

{document_text}

/no_think"""