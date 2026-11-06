# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:*11/6/2026*

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
 Record này nói một cả bức ảnh . bởi vì trong class_name là cab .
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp (class list)  đi kèm taxonomy/checkpoint 
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
vì trong 1 bức ảnh có nhiều vật thế khác nhau , có nhiều vật thế trùng nhau để không nhầm lẫn giữa các đối tượng .
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
để cho người gán nhãn không bị phân vân và đảm bảo tính nhất quát của dữ liệu .
6. Vì sao không được lấy `score` của model làm ground truth?
bởi vì score là kết quả do model tính toán ra , chưa chắc đã hoàn toán đúng nếu dùng làm ground truth sẽ ảnh hưởng tới model sau này
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`=[385.33,  69.24 , 498.92 , 348.92 ] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
góc trên-trái x= 385.33  y= 69.24 
góc dưới-phải x= 498.92  y= 348.92
rộng 113.58 pixel
cao 279.68 pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
0.20    17 vật thể
0.60    6 vật thể 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? 
lọc bỏ nhiều hộp dữ đoán sai hơn 
điều chỉnh và sửa đổi các hộp nhiều hơn 
xử lý các hộp bị trùng

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
khung phải ôm sát đối tượng , không để thừa phần nào ra     
chú ý những vẫn bị che khuất 
chú ý đối tượng nhỏ , nhiều
chú ý những đối tượng ở góc 

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
oven  bị che khuất ở bên trái người 
viết thành quy tắc guideline khi đó là quy định chung của dự án 
đưa lên hỏi lab coach (escalation) khi đó là trường hợp đặc biệt , mới , khác mà guideline hiện tại chưa bao phủ 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-006 ; `class_name`= bowl ; `score`= 0.528201 ; số điểm polygon=`polygon_point_count`= 84 ; 3–5 điểm đầu của`polygon_xy`= 
[29.0, 303.0] 
[28.0, 304.0]
[26.0, 304.0]
[25.0, 305.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
cung cấp thông tin chi tiết về đường biên cụ thể , độ lồi lõm và góc quay thực tế 
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dùng để định danh vật thể , 1 mã định danh riêng của nó 
class id là định danh của lớp
tracking id là định danh vật thể theo vế hoặc luồng thời gian 

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
mask vẽ ra phải ôm khít đối tượng , không thừa , không thiếu ,không cắt góc 

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
chỗ tay người ( person) và bowl 
mình sẽ dùng guideline trước nếu không có trong guideline mới escalation     
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một nhãn lớp cho toàn ảnh  | JSON phân loại, PNG có nhiều xe buýt và ô tô → khó chọn một lớp đại diện | Chọn nhãn theo bộ lớp và quy tắc đối tượng chính; đánh dấu trường hợp chưa rõ |  xem nhãn có phù hợp quy tắc hay không |
| Phát hiện vật thể |	Mỗi vật thể = lớp + bounding box   | khó đặt box cho người bị che  | Vẽ/sửa box từng vật thể; thêm vật thể thiếu, xóa dự đoán sai; xử lý che khuất theo quy định  | Đúng lớp, box ôm sát theo quy định; có bỏ sót, gán trùng hoặc bao nhầm vật thể không  |
| Instance segmentation | Mỗi cá thể = lớp + ID riêng + mask pixel/polygon;  |PNG chỉ thấy bó lá treo, không thấy chậu → nghi nhầm lớp | Xác minh lớp; vẽ/sửa mask sát phần nhìn thấy; tách riêng từng cá thể | Đúng lớp và ID; biên mask có ăn sang nền/vật khác, thiếu phần hoặc gộp nhầm cá thể không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Mình chỉ dùng dữ liệu được cấp trong phạm vi bài lab, không sao chép, chia sẻ hoặc tải lên dịch vụ bên ngoài khi chưa được phép 
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Mình sẽ dừng xử lý và báo cho giảng viên hoặc người phụ trách bài lab để được hướng dẫn 
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
