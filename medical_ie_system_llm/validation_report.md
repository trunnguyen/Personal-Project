# Validation Report

Checked 15 documents, 139 flagged items.

## position_text_mismatch (12 items)

- **doc 10** — `Căng thẳng nhiều trong công việc`
  raw_text[226:259] = 'Căng thẳng  nhiều trong công việc', but entity text = 'Căng thẳng nhiều trong công việc'
- **doc 10** — `đánh trống ngực: còn cảm giác đánh trống ngực khi nhập viện`
  raw_text[1053:1113] = 'đánh trống ngực:  còn cảm giác đánh trống ngực khi nhập viện', but entity text = 'đánh trống ngực: còn cảm giác đánh trống ngực khi nhập viện'
- **doc 10** — `Không buồn nôn, hay nôn, đổ mồ hôi`
  raw_text[1454:1489] = 'Không  buồn nôn, hay nôn, đổ mồ hôi', but entity text = 'Không buồn nôn, hay nôn, đổ mồ hôi'
- **doc 10** — `Ngày hôm nay khoảng Lúc 17 giờ, khi đang mang đồ tạp hóa ra xe, xuất hiện cảm giác thắt chặt ngực vùng trước tim, tăng đánh trống ngực, và khó thở kéo dài 20 giây`
  raw_text[1975:2138] = 'Ngày hôm nay khoảng Lúc 17 giờ,  khi đang mang đồ tạp hóa ra xe, xuất hiện cảm giác thắt chặt ngực vùng trước tim, tăng đánh trống ngực, và khó thở kéo dài 20 giây', but entity text = 'Ngày hôm nay khoảng Lúc 17 giờ, khi đang mang đồ tạp hóa ra xe, xuất hiện cảm giác thắt chặt ngực vùng trước tim, tăng đánh trống ngực, và khó thở kéo dài 20 giây'
- **doc 10** — `Được chỉ định điều trị aspirin 325mg x 1`
  raw_text[2175:2216] = 'Được chỉ định điều trị  aspirin 325mg x 1', but entity text = 'Được chỉ định điều trị aspirin 325mg x 1'
- **doc 12** — `có dịch giống mủ có màu vàng chảy ra từ tổn thương`
  raw_text[1507:1558] = 'có dịch giống mủ có màu vàng  chảy ra từ tổn thương', but entity text = 'có dịch giống mủ có màu vàng chảy ra từ tổn thương'
- **doc 18** — `****** (ổn định mảng xơ vữa)`
  raw_text[531:560] = '******  (ổn định mảng xơ vữa)', but entity text = '****** (ổn định mảng xơ vữa)'
- **doc 18** — `Triệu chứng liên quan: không thấy buồn nôn, nôn, ớn lạnh, thay đổi chức năng ruột, đau ngực, khó thở`
  raw_text[1570:1671] = 'Triệu chứng liên quan: không thấy  buồn nôn, nôn, ớn lạnh, thay đổi chức năng ruột, đau ngực, khó thở', but entity text = 'Triệu chứng liên quan: không thấy buồn nôn, nôn, ớn lạnh, thay đổi chức năng ruột, đau ngực, khó thở'
- **doc 18** — `Tự ý rời bệnh viện (AMA) ngày hôm qua sau khi cơn đau đỡ hơn ở lần nhập viện trước`
  raw_text[1712:1795] = 'Tự ý rời bệnh viện (AMA) ngày hôm qua sau khi cơn đau đỡ hơn ở  lần nhập viện trước', but entity text = 'Tự ý rời bệnh viện (AMA) ngày hôm qua sau khi cơn đau đỡ hơn ở lần nhập viện trước'
- **doc 20** — `Uống morphineoral`
  raw_text[1329:1347] = 'Uống  morphineoral', but entity text = 'Uống morphineoral'
- **doc 21** — `Bệnh nhân có đau lưng âm ỉ`
  raw_text[1697:1724] = 'Bệnh nhân có  đau lưng âm ỉ', but entity text = 'Bệnh nhân có đau lưng âm ỉ'
- **doc 21** — `Chụp cộng hưởng từ mật tụy ghi nhận sỏi đoạn cuối ống mật chủ`
  raw_text[2234:2296] = 'Chụp cộng hưởng từ mật tụy  ghi nhận sỏi đoạn cuối ống mật chủ', but entity text = 'Chụp cộng hưởng từ mật tụy ghi nhận sỏi đoạn cuối ống mật chủ'

## overlapping_spans (24 items)

- **doc 1** — `'đậu tằm' <-> 'đậu'`
  positions [1518, 1525] and [1518, 1521] overlap
- **doc 2** — `'sốt' <-> 'sốt cấp kéo dài'`
  positions [53, 56] and [53, 68] overlap
- **doc 2** — `'Mắt đỏ' <-> 'Mắt đỏ'`
  positions [1534, 1540] and [1534, 1540] overlap
- **doc 10** — `'Khó thở nhẹ khó thở' <-> 'Khó thở'`
  positions [688, 707] and [688, 695] overlap
- **doc 12** — `'Bệnh phổi kẽ do sử dụng corticoid liều cao kéo dài' <-> 'corticoid liều cao kéo dài'`
  positions [44, 94] and [68, 94] overlap
- **doc 12** — `'doxycycline' <-> 'doxycyclin'`
  positions [430, 441] and [430, 440] overlap
- **doc 13** — `'Lyssavirus' <-> 'virus'`
  positions [855, 865] and [860, 865] overlap
- **doc 14** — `'dấu mề đay' <-> 'mề đay'`
  positions [81, 91] and [85, 91] overlap
- **doc 15** — `'Bệnh phổi kẽ do sử dụng corticoid liều cao kéo dài' <-> 'corticoid liều cao kéo dài'`
  positions [44, 94] and [68, 94] overlap
- **doc 15** — `'Tình trạng tổn thương vùng âm hộ và mông bên phải ngày càng nặng' <-> 'mông bên phải'`
  positions [573, 637] and [609, 622] overlap
- **doc 16** — `'Lyssavirus' <-> 'virus'`
  positions [865, 875] and [870, 875] overlap
- **doc 18** — `'sốt nhẹ đến 38.3°C' <-> 'sốt'`
  positions [1113, 1131] and [1113, 1116] overlap
- **doc 18** — `'đau hạ sườn phải' <-> 'đau hạ sườn phải tái phát, ngày càng nặng hơn'`
  positions [1141, 1157] and [1141, 1186] overlap
- **doc 18** — `'đau hạ sườn phải liên tục' <-> 'đau hạ sườn phải liên tục'`
  positions [1408, 1433] and [1408, 1433] overlap
- **doc 19** — `'dấu mề đay' <-> 'mề đay'`
  positions [81, 91] and [85, 91] overlap
- **doc 19** — `'mề đay nổi rất khó chịu' <-> 'mề đay nổi rất khó chịu và ngứa'`
  positions [161, 184] and [161, 192] overlap
- **doc 19** — `'mề đay nổi rất khó chịu và ngứa' <-> 'mề đay nổi rất khó chịu'`
  positions [161, 192] and [161, 184] overlap
- **doc 19** — `'mề đay nổi rất khó chịu' <-> 'mề đay'`
  positions [161, 184] and [161, 167] overlap
- **doc 19** — `'MÀY ĐAY VÔ CĂN' <-> 'MÀY ĐAY VÔ CĂN'`
  positions [1161, 1175] and [1161, 1175] overlap
- **doc 19** — `'MÀY ĐAY MẠN TÍNH' <-> 'MÀY ĐAY MẠN TÍNH'`
  positions [1267, 1283] and [1267, 1283] overlap
- **doc 19** — `'MÀY ĐAY MẠN TÍNH' <-> 'MÀY ĐAY MẠN TÍNH'`
  positions [1267, 1283] and [1267, 1283] overlap
- **doc 19** — `'MÀY ĐAY MẠN TÍNH' <-> 'MÀY ĐAY MẠN TÍNH'`
  positions [1267, 1283] and [1267, 1283] overlap
- **doc 21** — `'Bệnh nhân có đau lưng âm ỉ' <-> 'đau lưng'`
  positions [1697, 1724] and [1711, 1719] overlap
- **doc 21** — `'Xét nghiệm chức năng gan ghi nhận tăng men gan' <-> 'tăng men gan'`
  positions [2449, 2495] and [2483, 2495] overlap

## exact_duplicate (7 items)

- **doc 2** — `Mắt đỏ`
  duplicate entity at [1534, 1540]
- **doc 18** — `đau hạ sườn phải liên tục`
  duplicate entity at [1408, 1433]
- **doc 19** — `mề đay nổi rất khó chịu`
  duplicate entity at [161, 184]
- **doc 19** — `MÀY ĐAY VÔ CĂN`
  duplicate entity at [1161, 1175]
- **doc 19** — `MÀY ĐAY MẠN TÍNH`
  duplicate entity at [1267, 1283]
- **doc 19** — `MÀY ĐAY MẠN TÍNH`
  duplicate entity at [1267, 1283]
- **doc 19** — `MÀY ĐAY MẠN TÍNH`
  duplicate entity at [1267, 1283]

## suspected_risk_factor_leak (5 items)

- **doc 2** — `hút thuốc lá thụ động`
  contains risk-factor keyword 'hút thuốc', typed TRIỆU_CHỨNG — should likely not be extracted at all
- **doc 10** — `Căng thẳng nhiều trong công việc`
  contains risk-factor keyword 'căng thẳng', typed TRIỆU_CHỨNG — should likely not be extracted at all
- **doc 10** — `Mất việc làm 8 ngày trước`
  contains risk-factor keyword 'mất việc', typed TRIỆU_CHỨNG — should likely not be extracted at all
- **doc 10** — `Một ngày người bệnh có thể uống hàng chục tách cà phê có caffeine`
  contains risk-factor keyword 'cà phê', typed TRIỆU_CHỨNG — should likely not be extracted at all
- **doc 17** — `Hút thuốc lá`
  contains risk-factor keyword 'hút thuốc', typed TRIỆU_CHỨNG — should likely not be extracted at all

## suspiciously_long_span (9 items)

- **doc 10** — `cảm giác thắt chặt ngực: Trung tâm, khởi phát lúc 17 giờ, kéo dài 20 giây, không có khó chịu vùng ngực khi đến tầng`
  115 chars (threshold 90 for TRIỆU_CHỨNG) — possible sentence-swallowing
- **doc 10** — `monitor holter cho thấy Nhịp xoang chiếm ưu thế. Ghi nhận ngoại tâm thu nhĩ và ngoại tâm thu thất xuất hiện thường xuyên`
  120 chars (threshold 90 for KẾT_QUẢ_XÉT_NGHIỆM) — possible sentence-swallowing
- **doc 10** — `Ngày hôm nay khoảng Lúc 17 giờ, khi đang mang đồ tạp hóa ra xe, xuất hiện cảm giác thắt chặt ngực vùng trước tim, tăng đánh trống ngực, và khó thở kéo dài 20 giây`
  162 chars (threshold 90 for TRIỆU_CHỨNG) — possible sentence-swallowing
- **doc 10** — `Viêm gan cấp tính do virus B thể thông thường điển hình mức độ nặng giai đoạn toàn phát`
  87 chars (threshold 60 for CHẨN_ĐOÁN) — possible sentence-swallowing
- **doc 10** — `monitor holter cho thấy Nhịp xoang chiếm ưu thế. Ghi nhận ngoại tâm thu nhĩ và ngoại tâm thu thất xuất hiện thường xuyên`
  120 chars (threshold 90 for KẾT_QUẢ_XÉT_NGHIỆM) — possible sentence-swallowing
- **doc 14** — `thể hệ sau không gây buồn ngủ là lựa chọn hàng đầu sau đó đến **************** thể hệ 1`
  100 chars (threshold 60 for THUỐC) — possible sentence-swallowing
- **doc 15** — `Lo ngại về Nhiễm virus Herpes simplex (HSV) hoặc Bệnh thủy đậu/Zona (do Varicella Zoster Virus)`
  95 chars (threshold 60 for CHẨN_ĐOÁN) — possible sentence-swallowing
- **doc 18** — `Triệu chứng liên quan: không thấy buồn nôn, nôn, ớn lạnh, thay đổi chức năng ruột, đau ngực, khó thở`
  100 chars (threshold 90 for TRIỆU_CHỨNG) — possible sentence-swallowing
- **doc 18** — `siêu âm vùng gan mật hiện tại cho thấy túi mật căng to với dịch quanh túi mật gợi ý viêm túi mật cấp`
  100 chars (threshold 90 for TRIỆU_CHỨNG) — possible sentence-swallowing

## suspected_procedure_as_drug (2 items)

- **doc 12** — `Sinh thiết nội mạc tử cung gần đây`
  contains procedure keyword 'sinh thiết' but typed THUỐC
- **doc 15** — `Sinh thiết nội mạc tử cung gần đây`
  contains procedure keyword 'sinh thiết' but typed THUỐC

## possible_missing_negation (7 items)

- **doc 1** — `sữa`
  negation cue 'không' appears earlier in the same clause, but isNegated not set
- **doc 13** — `dính nước dãi của chó con`
  negation cue 'chưa' appears earlier in the same clause, but isNegated not set
- **doc 13** — `vết cắn`
  negation cue 'không' appears earlier in the same clause, but isNegated not set
- **doc 13** — `điều trị y tế`
  negation cue 'không' appears earlier in the same clause, but isNegated not set
- **doc 16** — `dính nước dãi của chó con`
  negation cue 'chưa' appears earlier in the same clause, but isNegated not set
- **doc 20** — `dính nước dãi của chó con`
  negation cue 'chưa' appears earlier in the same clause, but isNegated not set
- **doc 100** — `tác dụng phụ của thuốc`
  negation cue 'không' appears earlier in the same clause, but isNegated not set

## empty_candidates (47 items)

- **doc 1** — `bệnh di truyền lặn liên kết với nhiễm sắc thể X`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `đậu tằm`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `đậu`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `Thuốc giảm đau, hạ sốt chứa *******`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `Kháng sinh nhóm ***********`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `Thuốc kháng sốt rét như *******, ***********`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `Vitamin K dùng trong điều trị nhiễm khuẩn tiết niệu`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `băng phiến`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `long não`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `thuốc nam`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `thuốc đông y`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 1** — `sữa`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `Điều trị quan trọng nhất`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `Dùng liều cao truyền tĩnh mạch`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `******* (ASA)`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `Dùng liều cao giai đoạn cấp tính`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `giảm liều duy trì để ngừa huyết khối`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `truyền **** lần 2`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `ức chế miễn dịch`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 2** — `Điều trị bằng **** trong 10 ngày đầu`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 10** — `được chỉ định siêu âm tim qua thành ngực vào tuần tới`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 13** — `họ Lyssaviridae`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `tăng lipid máu, không đặc hiệu`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `hẹp ống sống`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `Giả gout`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `bệnh thận mạn, không đặc hiệu Giai đoạn 4`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `tăng sản tuyến tiền liệt`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `Nhiều lần ngã gần đây`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `ảo giác`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 14** — `thể hệ sau không gây buồn ngủ là lựa chọn hàng đầu sau đó đến **************** thể hệ 1`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 16** — `Lyssavirus`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 16** — `họ Lyssaviridae`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 16** — `Phòng bệnh bằng tiêm *****************`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Viêm quanh răng`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Viêm nha chu`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `bệnh mạch máu ngoại biên`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `bệnh phổi tắc nghẽn mạn tính`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Ngưng thở khi ngủ do tắc nghẽn đang dùng BiPAP`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Ung thư biểu mô tế bào vảy xâm nhập của dương vật`
  CHẨN_ĐOÁN entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Uống trà gừng và mật ong`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Trà đinh hương`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `bào láng gốc răng`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 17** — `Ghép mô mềm ở vòm họng`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 18** — `****** (ổn định mảng xơ vữa)`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 18** — `***************** / ***`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 20** — `alevenhưng`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term
- **doc 21** — `Đã ngừng sử dụng thuốc NSAIDs`
  THUỐC entity has no candidates — possible retrieval miss or bad lookup_term

## possible_missing_family (26 items)

- **doc 11** — `mất định hướng`
  family cue 'người nhà' appears earlier in the same clause, but isFamily not set
- **doc 13** — `bị thương nhẹ ở tay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 13** — `bị chảy máu`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 13** — `chó con`
  family cue 'cha' appears earlier in the same clause, but isFamily not set
- **doc 13** — `chó chưa tiêm dại`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 13** — `dính nước dãi của chó con`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 13** — `được`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 14** — `dấu mề đay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 14** — `mề đay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 16** — `bị thương nhẹ ở tay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 16** — `bị chảy máu`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 16** — `chó chưa tiêm dại`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 16** — `dính nước dãi của chó con`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 17** — `đánh răng hay chảy máu chân răng`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 17** — `miệng thấy hơi thở mùi khó chịu`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 19** — `dấu mề đay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 19** — `mề đay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 20** — `bị thương nhẹ ở tay`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 20** — `bị chảy máu`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 20** — `chó con`
  family cue 'cha' appears earlier in the same clause, but isFamily not set
- **doc 20** — `dính nước dãi của chó con`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 21** — `Rối loạn chuyển hóa tinh bột (amyloidosis)`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 100** — `mang thai được 22 tuần`
  family cue 'cha' appears earlier in the same clause, but isFamily not set
- **doc 100** — `cục máu đông`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 100** — `đi tiêu ra máu`
  family cue 'em' appears earlier in the same clause, but isFamily not set
- **doc 100** — `tác dụng phụ của thuốc`
  family cue 'em' appears earlier in the same clause, but isFamily not set

