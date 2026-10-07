# Tests

เก็บโค้ด unit test และ regression test ของโปรเจกต์ไว้ในโฟลเดอร์นี้

- `test_track_evaluation_stability.py`: ตรวจ Nogueira, Kuncheva, feature universe และการบันทึกผลของ `04_track_evaluation.py`
- `test_track_evaluation_shap.py`: ตรวจ SHAP ของ KNN/NB เทียบกับสูตร Shapley แบบไล่ครบทุกชุดบนข้อมูลจำลองขนาดเล็ก, คลาส Fail, seed, additivity, สถานะล้มเหลว และการบันทึก JSON ผ่าน loop จริง
- `test_visualize_result.py`: ตรวจกราฟ basic ใน `05_visualize_result.py` ครอบคลุมค่าที่อ่านจาก JSON, การแบ่งหน้ากราฟ, ค่า NA, ชื่อยาว และความเหมือนกันของผลเมื่อรันซ้ำ

รันจากโฟลเดอร์หลักของโปรเจกต์:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

รันเฉพาะชุด:

```sh
.venv/bin/python -m unittest tests.test_track_evaluation_stability -v
.venv/bin/python -m unittest tests.test_track_evaluation_shap -v
.venv/bin/python -m unittest tests.test_visualize_result -v
```

Tests ไม่รันการทดลองจริงและไม่แก้ JSON การทดลองเดิม ชุด SHAP ฝึกโมเดลด้วยข้อมูลจำลองขนาดเล็กเท่านั้น ผลจำลองถูกเขียนลงโฟลเดอร์ชั่วคราวและล้างเมื่อจบการทดสอบ ส่วนการเทียบกับผลการทดลองจริงจะอ่านไฟล์ใน `experiments/` และรายงานวินิจฉัยใน `audits/` ที่ระบุไว้ใน test

SHAP ของ KNN/NB อธิบายความน่าจะเป็นของ Fail (label=1) ด้วย PermutationExplainer และ background จาก training fold ก่อน resampling ทุกแถวใน validation fold ถูกอธิบาย ค่าเป็นการประมาณจาก permutation; การผ่าน additivity ไม่ได้ยืนยันว่าค่าแต่ละ feature หรืออันดับลู่เข้าแล้ว งบเริ่มต้นคือ `20 * (2 * n_features + 1)` evaluations ต่อแถว จึงอาจช้ามากเมื่อมีหลาย feature ห้ามเทียบขนาด SHAP ในหน่วย probability กับ log-odds โดยตรง

ผลสำเร็จใช้ schema เดิมใน `<run_id>_shap.json` ส่วน `<run_id>_shap_status.json` บันทึกสถานะและสาเหตุเมื่อคำนวณไม่ได้ โดยไม่เติมศูนย์แทน และเก็บ prediction metrics ตามเดิม พารามิเตอร์และหน่วยผลลัพธ์อยู่ใน `<run_id>_config.json` หากนำผลไปสรุปข้าม fold ต้องตรวจจำนวนสถานะที่สำเร็จด้วย

Visualization ใช้เฉพาะ basic: metrics เฉลี่ย, feature stability และจำนวน feature เฉลี่ย โดยอ่านค่าที่บันทึกไว้ใน JSON เท่านั้น ไม่มีโหมด paper/RQ1–RQ5 หรือชุดทดสอบที่ฝังอยู่ในไฟล์หลัก
