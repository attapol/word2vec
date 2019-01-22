# word2vec : Metonymy of countries in ไทยรัฐ

## 1. Method
หา correspondence ระหว่างประเทศกับนามนัย (metonymy) โดยใช้ deep learning (word2vec) <br>
จุดประสงค์มีสองข้อ
* แสวงหาวิธีค้นหา metonymy โดยอัตโนมัติ เพื่อประยุกต์กับ NLP
* หา prototypical metonymy 
เก็บข้อมูลมาจาก "ไทยรัฐ" ทั้งหมด 362985 บทความ (18/01/2562) <br>
ใช้ ทั้ง CBOW และ skip-gram model แล้วเปรียบเทียบกัน (ฝึกโดยเนื้อหาบทความเท่านั้น)

### 1.1 python toolkit
tokenizer: `pythainlp.tokenize.word_tokenize` <br>
word2vec: `gensim.model.word2vec` sg=0,1(CBOW, skip-gram), size=200 (มิติ), min_count=5, window=15 <br>

### 1.2 cosine similarity 
วัดความคล้ายคลึงของสอง vector โดยใช้ cosine similarity <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" title="\cos{\theta} = \frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" /></a> <br>
ในปริภูมิ 200 มิติ cosine similarity ของสอง vector ที่สุ่มเลือกมา ส่วนใหญ่จะต่ำกว่า 0.5 ( -> [random_vector.py](https://github.com/nozomiyamada/word2vec/blob/master/random_vector.py) ) `sim_distribution`<br>
![cos_sim_dis_2](https://user-images.githubusercontent.com/44984892/51155083-e78f2080-18a8-11e9-8800-c6f94f898597.png)
![cos_sim_dis_3](https://user-images.githubusercontent.com/44984892/51155084-e827b700-18a8-11e9-8b3f-1b8434bf3fdd.png)
![cos_sim_dis_50](https://user-images.githubusercontent.com/44984892/51155085-e827b700-18a8-11e9-9b0d-0dd3d347a9fa.png)
![cos_sim_dis_200](https://user-images.githubusercontent.com/44984892/51155087-e8c04d80-18a8-11e9-97e3-91fe07f0f67f.png)

|มิติ |จำนวนที่ similarity สูงกว่า 0.5 |ความน่าจะเป็น | |
|:-:|--:|--:|:--|
|2 |166215 / 499500 |33.28% |[-π/3, π/3] / 2π = 1/3 |
|3 |124860 / 499500 |25.00% |1/2 * 2π / 4π sr = 1/4 |
|50 |50 / 499500 |0.01% | |
|200 |0 / 499500 |< 0.0002% | |

เพราะฉะนั้น ในปริภูมิ 200 มิติ สามารถคิดได้ว่า "มี similarity **ตั้ง** 0.5" แทนที่จะคิดว่า "มี similarity **แค่** 0.5" (เกณฑ์ similarity ไม่เหมือนกับกรณี 2, 3 มิติที่เราคุ้นเคย )<br>
แต่ต้องคำนึงถึง distribution จริงเป็นยังไง

### 1.3 distribution
การกระจายของ metonyms ในปริภูมิเวกเตอร์จะแสดงให้เห็นว่า metonym แบบไหนเป็น prototype และ metonym แบบไหนเป็น peripheral <br>
นอกจากนี้ เพื่อที่จะวิเคราะห์ความสัมพันธ์ระหว่างประเทศกับนามนัย การวัด distance อาจจะมีประโยชน์ แต่ต้องเลือก distance ที่เหมาะสม

## 2. Result: similarity & distance of word pair
<table>
<tr>
  <th>CBOW</th>
  <th>skip-gram</th>
</tr>
<tr>
  <td>
    <pre>
met1.similar('ปลาดิบ')
อาทิตย์อุทัย 0.8377
กิมจิ 0.7883
จิงโจ้ 0.7115
ลอดช่อง 0.6982
อิเหนา 0.6715
หมีขาว 0.6469
ซามูไร 0.6372
มะกะโรนี 0.6058
ฝอยทอง 0.536
โสมขาว 0.5214
จังโก้ 0.4983
มังกร 0.4919
ผู้ดี 0.4893
กีวี 0.4826
น้ำหอม 0.4716
ไก่งวง 0.4701
ผีดิบ 0.4376
กระทิง 0.4324
มัตสึ 0.4323
เบค" 0.4142
    </pre>
  </td>
  <td>
    <pre>
met2.similar('ปลาดิบ')
อาทิตย์อุทัย 0.7143
กิมจิ 0.6961
ซามูไร 0.6283
ญี่ปุ่น 0.6203
ประเทศญี่ปุ่น 0.6098
โอกะ 0.5709
Tinydoll 0.5689
โตะ 0.5666
เกะ 0.562
ดะ 0.559
สึ 0.5479
มัตสึ 0.5467
ซุ 0.5442
โตะ 0.5436
กิ 0.5391
จิงโจ้ 0.5359
ชาวญี่ปุ่น 0.5324
โสมขาว 0.5323
คัสซึโตะ 0.5303
ลอดช่อง 0.5276
    </pre>
  </td>
</tr>
<tr>
  <td>
    <pre>
met1.similar('มังกร', 10)
จิงโจ้ 0.5143
ปลาดิบ 0.4919
อาทิตย์อุทัย 0.4899
หมีขาว 0.4875
ฝอยทอง 0.4816
ซามูไร 0.4778
อิเหนา 0.4723
ขุนแผน 0.4599
ค้างคาว 0.4488
อสูร 0.4483
    </pre>
  </td>
  <td>
    <pre>
met2.similar('มังกร', 10)
หงส์ 0.5329
หยก 0.5165
สิงโต 0.5101
วานร 0.5093
กว่าง 0.4947
ส.อู๊ดดี้ 0.4928
ฟ้า 0.4891
โปลิศ 0.4846
ฉวน 0.4791
ราชสีห์ 0.4759
    </pre>
  </td>
</tr>
<tr>
  <td>
    <pre>
met1.similar('ผู้ชาย', 10)
ผู้หญิง 0.9242
กะเทย 0.6003
คนเจ้าชู้ 0.6003
เด็กผู้หญิง 0.5948
เด็กผู้ชาย 0.5924
เลสเบี้ยน 0.5855
ชายหนุ่ม 0.5773
เกย์ 0.564
เจ้าชู้ 0.5586
สำส่อน 0.5565
    </pre>
  </td>
  <td>
    <pre>
met2.similar('ผู้ชาย', 10)
ผู้หญิง 0.934
คนเจ้าชู้ 0.6667
เจ้าชู้ 0.6653
คบ 0.6608
เด็กผู้ชาย 0.6581
คนอื่น 0.647
กะเทย 0.6453
จีบ 0.6446
เค้า 0.6412
ชายหนุ่ม 0.6374
    </pre>
  </td>
</tr>
</table>

### cosine similarity distribution of random two word pair (CBOW & skip-gram)

![sim_distribution_cbow](https://user-images.githubusercontent.com/44984892/51425715-5b675b00-1c12-11e9-8ce6-dd592632a4e8.png)
![sim_distribution_skip](https://user-images.githubusercontent.com/44984892/51425749-c6b12d00-1c12-11e9-8a71-df22f8f19894.png)

ในกรณี skip-gram ค่ามัฌชิมไม่ใช่ 0 แสดงว่า vector เหล่านี้เป็น uneven distribution ในปริภูมิ 200 มิติ

### distance distribution of random two word pair (CBOW & skip-gram)

![dis_distribution_cbow](https://user-images.githubusercontent.com/44984892/51425985-67edb280-1c16-11e9-95d7-3363d16a0077.png)
![dis_distribution_skip](https://user-images.githubusercontent.com/44984892/51425987-67edb280-1c16-11e9-995d-fd9d715671ad.png)

สามารถสังเกตได้ว่า skip-gram ให้ vector เบียดกว่าในปริภูมิ 200 มิติ

|word1 | word2 | similarity (CBOW) | similarity (skip-gram) | distance (CBOW) | distance (skip-gram) |
|:-:|:-:|--:|--:|--:|--:|
|สวย|โรงเรียน|-0.0522| 0.1545 |58.1469 | 3.7336 |
|ไป |อร่อย |0.0400 |0.1756 |54.2978|3.8373 |
|ครับ |จุฬา |-0.0142 | 0.0739 |56.6979 | 4.6446 |
|การ์ตูน |หนังสือ |0.2062 | 0.3472 | 50.9731 | 3.8464 |
|สยาม |พารากอน |0.3081 | 0.3749 | 39.4318| 4.1475 |

## 3. Result: vector calculation
### 3.1 สมมติฐาน 1: metonymy is a parallel translation
![met_vector](https://user-images.githubusercontent.com/44984892/51070601-7ff18f00-1676-11e9-809e-eda1ae81a817.jpg) <br>
#### 3.1.1 mean metonymization vector 
<a href="https://www.codecogs.com/eqnedit.php?latex=\overrightarrow{f}&space;=&space;\frac{1}{n}&space;\sum_{X}\overrightarrow{XX'}&space;=&space;\frac{1}{n}\left(&space;(\overrightarrow{OA'}&space;-&space;\overrightarrow{OA})&space;&plus;&space;(\overrightarrow{OB'}&space;-&space;\overrightarrow{OB})&space;&plus;&space;\cdots&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\overrightarrow{f}&space;=&space;\frac{1}{n}&space;\sum_{X}\overrightarrow{XX'}&space;=&space;\frac{1}{n}\left(&space;(\overrightarrow{OA'}&space;-&space;\overrightarrow{OA})&space;&plus;&space;(\overrightarrow{OB'}&space;-&space;\overrightarrow{OB})&space;&plus;&space;\cdots&space;\right)" title="\overrightarrow{f} = \frac{1}{n} \sum_{X}\overrightarrow{XX'} = \frac{1}{n}\left( (\overrightarrow{OA'} - \overrightarrow{OA}) + (\overrightarrow{OB'} - \overrightarrow{OB}) + \cdots \right)" /></a> <br>
where X: country, X': metonymy of country

#### 3.1.2 วิธีค้นหา X หรือ X'
1. (country B) + (metonymy A') - (country A) = (metonymy B') : country B + metonymization vector
2. (metonymy B') + (country A) - (metonymy A') = (country B) : metonymy B' - metonymization vector

##### 3.1.2.1 วิธี 1 ค้นหา metonym
|สูตร|CBOW |Skip-gram |
|:-:|:-:|:-:|
| 'เกาหลีใต้' + ปลาดิบ' - 'ญี่ปุ่น' | กิมจิ 0.6943<br>โสมขาว 0.6832<br>อาทิตย์อุทัย 0.6401 ... | ('โสมขาว', 0.7123)<br>('กิมจิ', 0.6929)<br>('โสมแดง', 0.5844) ... |
| 'อังกฤษ' + 'ซามูไร' - 'ญี่ปุ่น' | ผู้ดี 0.6433<br>ส์" 0.5827<br>เวลส์ 0.5817 ... | ('ผู้ดี', 0.6484)<br>('พรีเมียร์', 0.6449)<br>('คำราม', 0.6180) ... |
| 'โปรตุเกส' + 'มังกร' - 'จีน' | ('ฝอยทอง ', 0.5421)<br>('เวลส์', 0.4716)<br>('สแปนิช', 0.4350) ... | ('ราชัน', 0.5837)<br>('ฝอย', 0.5761)<br>('ฝอยทอง', 0.5653) ... |
| 'อิตาลี' + 'มังกร' - 'จีน' | ('มะกะโรนี', 0.5553)<br>('สเปน', 0.4701)<br>('เอซี มิลาน', 0.4484) ... | ('กัลโช', 0.5421)<br>('มะกะโรนี', 0.5274)<br>('เอซี มิลาน', 0.5259) ... |

##### 3.1.2.2 วิธี 2 ค้นหา country
|สูตร|CBOW |Skip-gram |
|:-:|:-:|:-:|
|'มะกะโรนี' + 'ญี่ปุ่น' - 'ปลาดิบ' | ('อิตาลี', 0.5934)<br>('อังกฤษ', 0.5056)<br>('เอซี มิลาน', 0.4917) ... |('อิตาลี', 0.7146)<br>('อังกฤษ', 0.6199)<br>('ตุรกี', 0.5956) ... |
|'จิงโจ้' + 'เกาหลีใต้' - 'กิมจิ' | ('ไต้หวัน', 0.5936)<br>('ออสเตรเลีย', 0.5897)<br>('อินโดนีเซีย', 0.57834) ... | ('ออสเตรเลีย', 0.6735)<br>('เคนยา', 0.5914)<br>('เปรู', 0.5903) ... |
|'กีวี' + 'จีน' - 'มังกร' | ('นิวซีแลนด์', 0.5327)<br>('ญี่ปุ่น', 0.5150)<br>('ฮ่องกง', 0.5019) ... | ('นิวซีแลนด์', 0.6551)<br>('ออสเตรเลีย', 0.5848)<br>('ไต้หวัน', 0.5774) ... |
|'อิเหนา' + 'ญี่ปุ่น' - 'ซามูไร' | ('สิงคโปร์', 0.6267)<br>('เวียดนาม', 0.6051)<br>('อินโดนีเซีย', 0.6032) ... | ('อินโดนีเซีย', 0.7172)<br>('สิงคโปร์', 0.6183)<br>('ฟิลิปปินส์', 0.5986) ... | 
|'กระทิง' + 'เกาหลีใต้' - 'กิมจิ' |('สเปน', 0.5052)<br>('เยอรมนี', 0.4368)<br>('อิตาลี', 0.4238) ... |('สเปน', 0.5557)<br>('บราซิล', 0.4902)<br>('อุรุกวัย', 0.4894) ... |
|'น้ำหอม' + 'เกาหลีใต้' - 'กิมจิ' | ('เยอรมนี', 0.5452)<br>('ฝรั่งเศส', 0.5285)<br>('เนเธอร์แลนด์', 0.4940) ... |('ฝรั่งเศส', 0.6040)<br>('บราซิล', 0.5664)<br>('ปารีส', 0.5271) ... |

#### 3.1.3 Explanation by classical semantics (semantic features) 
|คำ |ความเป็นนามนัย |ความเป็นญี่ปุ่น |ความเป็นเกาหลี | 
|:-:|:-:|:-:|:-:|
|ญี่ปุ่น  | - | + | - |
|ปลาดิบ  | + | + | - |
|เกาหลี  | - | - | + |
|กิมจิ  | + | - | + |
|||||
|เกาหลี + ปลาดิบ | + | + | + |
|เกาหลี + ปลาดิบ - ญี่ปุ่น| + | - | + |

#### 3.1.4 similarity of each metonymization vector
![metonymy_table](https://user-images.githubusercontent.com/44984892/51151876-0b4d6900-189f-11e9-9159-1b86f5892305.png)

### 3.2 สมมติฐาน 2: metonymy is Affine Transformation
<a href="https://www.codecogs.com/eqnedit.php?latex=\vec{x'}&space;=&space;A\vec{x}&space;&plus;&space;\vec{b}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{x'}&space;=&space;A\vec{x}&space;&plus;&space;\vec{b}" title="\vec{x'} = A\vec{x} + \vec{b}" /></a> <br>
ถ้าสมมติ Affine Transformation ต้องหา linear transformation matrix A กับ translation vector b <br>
ในกรณีนี้ vector มี 200 มิติ สัมประสิทธิ์ก็มีทั้งหมด 200^2 (A) + 200 (b) = 40200 ตัว (ต้องการ 201 vector ที่ต่างกัน) แต่เก็บตัวอย่างมากขนาดนี้ไม่ได้ เพราะฉะนั้นต้องการ regression model อะไรสักอย่าง เช่น least squares นอกจากนั้น ลองทำ approximation เป็น diagonal matrix

#### 3.2.1 Affine Transformation with diagonal matrix
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&0&\cdots&0&b_1\\&space;0&A_{2,2}&\cdots&0&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;0&0&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&0&\cdots&0&b_1\\&space;0&A_{2,2}&\cdots&0&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;0&0&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" title="\begin{pmatrix} x_1'\\x_2'\\\vdots\\x_{200}'\\1 \end{pmatrix} = \begin{pmatrix} A_{1,1}&0&\cdots&0&b_1\\ 0&A_{2,2}&\cdots&0& b_2\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&\cdots&A_{200,200}&b_{200}\\ 0&0&\cdots&0&1 \end{pmatrix} \begin{pmatrix} x_1\\x_2\\\vdots\\x_{200}\\1 \end{pmatrix}" /></a><br>
ในกรณีนี้ คำนวณแต่ละสัมประสิทธิ์ได้โดย simple linear regression <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=x_i'=A_{i,i}x_i&plus;b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i'=A_{i,i}x_i&plus;b_i" title="x_i'=A_{i,i}x_i+b_i" /></a>

Result (skip-gram): detA (product of eigenvalue A<sub>i,i</sub>) = -4.824009094491605e-46 <br>
cosine similarity of "mean metonymization vector" and "parallel translation b" = 0.8057
ในกรณีนี้ ไม่มีค่าใดที่มากกว่า 1 ใน A เพราะฉะนั้น สามารถสรุปได้ว่า metonymization ไม่ใช่ parallel translation อย่างเดียว แต่ similarity ยังมี 0.8 metonymization จึงเกิดจาก parallel translation b เป็นหลัก

#### 3.2.2 Affine Transformation with full matrix
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\&space;A_{2,1}&A_{2,2}&\cdots&A_{2,200}&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\&space;A_{2,1}&A_{2,2}&\cdots&A_{2,200}&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" title="\begin{pmatrix} x_1'\\x_2'\\\vdots\\x_{200}'\\1 \end{pmatrix} = \begin{pmatrix} A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\ A_{2,1}&A_{2,2}&\cdots&A_{2,200}& b_2\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\ 0&0&\cdots&0&1 \end{pmatrix} \begin{pmatrix} x_1\\x_2\\\vdots\\x_{200}\\1 \end{pmatrix}" /></a><br>
ในกรณีนี้ คำนวณแต่ละสัมประสิทธิ์ได้โดย multiple linear regression <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=x_i'=A_{i,1}x_1&plus;A_{i,2}x_2&plus;\cdots&plus;A_{i,200}x_{200}&plus;b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i'=A_{i,1}x_1&plus;A_{i,2}x_2&plus;\cdots&plus;A_{i,200}x_{200}&plus;b_i" title="x_i'=A_{i,1}x_1+A_{i,2}x_2+\cdots+A_{i,200}x_{200}+b_i" /></a>

Result (skip-gram): detA ~ 0 <br>
cosine similarity of "mean metonymization vector" and "parallel translation b" = -0.14217
อิทธิพลจาก parallel translation b น้อยลง เพราะฉะนั้น ถ้าใช้ full matrix A อาจจะค้นหา metonyms ได้ดีกว่า <br>

ผลลัพธ์ที่เปรียบเทียบ "country + mean metonymization vector" กับ "full matrix Affine Transfomation" เป็นดังนี้

|country : metonym | + mean metonymization vector | Affine Transformation (diag matrix) | Affine Transformation (full matrix) |
|:-:|:-:|:-:|:-:|
|ญี่ปุ่น : ปลาดิบ | 0.814376260782165 | 0.8309753576830674 | 0.758601779183857 |
|ญี่ปุ่น : ซามูไร | 0.7398626506470664 | 0.7523959310138825 | 0.823747168402788 |
|ญี่ปุ่น : นินจา | 0.4485568436522527 |0.44680430519498154 | 0.7408583027819111|
|ญี่ปุ่น : อาทิตย์อุทัย |0.7926451916486672 | 0.8039604948178302 | 0.7973922129586257 |
|จีน : มังกร |0.6448982167504311 | 0.6788160962580067 | 0.9996140535040609 |
|เกาหลี : กิมจิ |0.7844103622475953 |0.8175731068442497 | 0.9991522213997927 |
|เกาหลีเหนือ : โสมแดง |0.8672777197761677 | 0.8670044449078599 | 0.9993399396725856 |
|เกาหลีใต้ : โสมขาว |0.8487338701615814 | 0.8537188826579071 | 0.9992611952491016 |
|อังกฤษ : ผู้ดี |0.8062117758554058 | 0.8228773867761392 | 0.9984084064099905 |
|ตุรกี : ไก่งวง |0.6188068272787861 | 0.6554135834169289 | 0.999741675968432 |
|ออสเตรเลีย : จิงโจ้ |0.7708969496916427 | 0.7887867506106588 | 0.9996214946014829 |

mean metonymization vector กับ diagonal matrix Affine transformation ไม่ค่อยต่างกัน ส่วน full matrix Affine transformation ในกรณี 'ญี่ปุ่น' ข้อมูลที่ใช้ไม่ใช่ one-to-one mapping เพราะฉะนั้น similarity ก็ประมาณ 0.7 - 0.8 แต่ถ้า metonymy เป็น one-to-one mapping ก็จะได้ similarity ที่มากกว่า 0.99 
แต่พอใช้ข้อมูลที่ไม่อยู่ใน list ได้ผลที่ไม่ดี

|country : metonym | mean metonymization vector | Affine Transformation (diag matrix) | Affine Transformation (full matrix) |
|:-:|:-:|:-:|:-:|
|บราซิล : แซมบ้า|0.8594228910356522 | 0.8499922994206792 | -0.12161083401520253|
|อียิปต์ : มัมมี่ | 0.6953887445803948 | 0.6617337737362334 |  -0.09990733670975752 |
|เยอรมัน : เบียร์ | 0.49053095957711806 | 0.4903473733385831 | 0.13036258688194893 |
|เยอรมัน : ม้าลาย | 0.3780971713257477 | 0.43227006559087516 | 0.08425966845519557 |
|อเมริกา : นกอินทรี |0.3451728274883405 | 0.3868838186755263 | 0.0802828885144199 |
|จีน : แผ่นดินใหญ่ | 0.6420330389110807 | 0.6237732370603349 | 0.3787957753457047 |
|ไทย : ช่าง | 0.2007920627510823 |0.22125964375458845 |0.09029290403054076 |

ผลลัพธ์ของ full matrix คือ  **overfitting** เพราะ similarity เกือบ 0 (mean ~ 0.19) <br>
เพราะฉะนั้น การใช้แค่ parallel translation vector อย่างเดียว หรือ diagonal matrix Affine transformation น่าจะสามารถ detect metonym ได้ดีกว่า แต่บาง metonym เช่น ไทย : ช้าง ทำอย่างไรก็ค้นไม่เจอ

### 3.3 Distance

#### 3.3.1 Distance between country and metonymy
* Euclidean Distance -> แต่ละมิติต้องเป็น orthogonal basis แต่ word2vec ไม่เหมือนกับ SVD เพราะฉะนั้น ความชัดเจนน้อยลง ถ้าสุ่มเลือกสองจุดในปริภูมิ 200 มิติโดย Gaussian distribution แล้ว distribution ของ Euclidean distance ของสองจุดนี้ก็จะเป็น Gaussian เหมือนกัน โดยมีค่ามัชฌิม 20 ( -> [random_vector.py](https://github.com/nozomiyamada/word2vec/blob/master/random_vector.py) ) `dis_distribution`<br>
![dis_distribution_random](https://user-images.githubusercontent.com/44984892/51271873-419ffb00-19fb-11e9-9937-313c2e138307.png) <br>

ผลลัพธ์ : Euclidean distance between country and metonymy <br>
<img src="https://user-images.githubusercontent.com/44984892/51412426-202b4480-1b9e-11e9-98de-154b4b1435c2.png" width="600px" >

distance ของ metonymization vector : mean 3.2456
* Wasserstein Embeddings 

#### 3.3.2 Distance among metonyms (or countries)

* ~~Mahalanobis Distance -> เพื่อเปรียบเทียบทั้ง metonymy และสามารถหา prototype ที่มีความนามนัยสูงที่สุดได้ แต่ต้องสมมติการกระจายเป็น Gaussian~~ <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=N(\vec{x}|\vec{\mu},\Sigma)&space;=&space;\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}&space;\exp&space;\left[&space;-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(\vec{x}|\vec{\mu},\Sigma)&space;=&space;\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}&space;\exp&space;\left[&space;-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})&space;\right]" title="N(\vec{x}|\vec{\mu},\Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} \exp \left[ -\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu}) \right]" /></a> <br>
ใช้ไม่ได้ เพราะต้องเป็นแบบ **จำนวนข้อมูล > จำนวนมิติ (full rank)** ไม่อย่างนั้น หา inverse matrix ของ covariance matrix ไม่ได้

* k-means : clustering metonyms into subtypes 

* Poincare Embeddings

### 3.4 Canonical Correlation Analysis

## 4. รายชื่อ metonym ที่พบเจอใน "ไทยรัฐ"
|ประเทศ  |นามนัย |
|:-:|:--|
|ญี่ปุ่น  |ปลาดิบ, ซามูไร, นินจา, อาทิตย์อุทัย |
|จีน  |มังกร, แผ่นดินใหญ่  |
|เกาหลี |กิมจิ, โสมแดง(เหนือ), โสมขาว(ใต้) |
|อังกฤษ |ผู้ดี |
|ตุรกี |ไก่งวง |
|ออสเตรเลีย |จิงโจ้ |
|นิวซีแลนด์ |กีวี |
|กัมพูชา |นครวัด |
|อินโดนีเซีย |อิเหนา |
|อีนเดีย |ภารตะ |
|มองโกเลีย |เจงกิสข่าน |
|สิงคโปร์ |ลอดช่อง, เมอร์ไลออน |
|อียิปต์ |มัมมี่ |
|สหรัฐฯ |นกอินทรี, แฮมเบอร์เกอร์ |
|อิตาลี |มะกะโรนี |
|รัสเซีย |หมีขาว |
|ฝรั่งเศส |น้ำหอม |
|เยอรมัน |เบียร์ |
|สเปน |กระทิง |
|โปรตุเกส |ฝอยทอง |
|เนเธอร์แลนด์ |กังหันลม | 

### 4.1 distribution of metonyms

3D visualiztion (โดยใช้ PCA)

![distribution_3d_country](https://user-images.githubusercontent.com/44984892/51482080-0ec58080-1dc8-11e9-9d8e-57159022792c.png)
![distribution_3d_metonymy](https://user-images.githubusercontent.com/44984892/51482078-0ec58080-1dc8-11e9-833b-acc8477deb7b.png)

## 5. Problems
* คำศัพท์บางคำหาไม่เจอโดยใช้วิธีนี้ เช่น คำว่า 'ช้าง' ซึ่งเป็นนามนัยของประเทศไทย อาจจะเป็นเพราะคำว่าข้างใช้ในบริบทธรรมดาด้วย ส่วนคำว่า 'ปลาดิบ' ไม่ค่อยปรากฏในความหมายดั้งเดิม (ปรากฏเฉพาะในกรณีพูดถึงประเทศญี่ปุ่น)
* พูดถึง metonymy แล้ว ควรวิเคราะห์เชิง coginitive linguistics แต่ไม่รู้ว่าจะเชื่อมโยงกันได้อย่างไง ยกเว้นแนวคิด prototype
* tokenizer ตัดคำไม่ถูก เช่น "นินจา" จะเป็น นิ-น-จา แต่ PyThaiNLP ไม่มี method ที่เพิ่มคำศัพท์ เพราะฉะนั้น ตัดไปแล้วค่อยเอามารวมกันอีกโดยใช้ unix `sed`

## 6. Extra
* 'จีน' + 'โตเกียว' - 'ญี่ปุ่น' = ('ปักกิ่ง', 0.7812713384628296) ...
* 'จีน' + 'เกียวโต' - 'ญี่ปุ่น' = ('ฉงชิ่ง', 0.6020291447639465) ...
* 'ต้มยำกุ้ง' + 'ญี่ปุ่น' - 'ไทย' = ('แซลมอน', 0.4649949371814728) ...
* 'ฮ่องเต้' + 'ญี่ปุ่น' - 'จีน' = ('จักรพรรดิ', 0.5973250865936279) ...
* 'ภรรยา' - 'สาว' = ('บุตรชาย', 0.49089011549949646) ...
* 'ประวิตร' - 'นาฬิกา' = ('ประยุทธ์', 0.5291631817817688) ... จริงๆ ไม่ต้องลบ 'นาฬิกา' ก็คล้ายกันอยู่แล้ว
