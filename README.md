# word2vec : Metonymies of countries in ไทยรัฐ

## 1. Method
หา correspondence ระหว่างประเทศกับนามนัย (metonymy) โดยใช้ deep learning (word2vec) <br>
เก็บข้อมูลมาจาก "ไทยรัฐ" ทั้งหมด 203624 บทความ (12/01/2562) <br>
ใช้ CBOW model (ฝึกโดยเนื้อหาบทความเท่านั้น)

### 1.1 python toolkit
tokenizer: `pythainlp.tokenize.word_tokenize` <br>
word2vec: `gensim.model.word2vec` (sg=0, size=200, min_count=5, window=15) <br>

### 1.2 cosine similarity 
วัดความคล้ายคลึงโดยใช้ cosine similarity <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" title="\cos{\theta} = \frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" /></a> <br>
ในปริภูมิ 200 มิติ cosine similarity ของสอง vector ที่สุ่มเลือกมา ส่วนใหญ่จะต่ำกว่า 0.5 ( -> [random_vector.py](https://github.com/nozomiyamada/word2vec/blob/master/random_vector.py) ) `sim_distribution`<br>
![cos_sim_dis_2](https://user-images.githubusercontent.com/44984892/51155083-e78f2080-18a8-11e9-8800-c6f94f898597.png)
![cos_sim_dis_3](https://user-images.githubusercontent.com/44984892/51155084-e827b700-18a8-11e9-8b3f-1b8434bf3fdd.png)
![cos_sim_dis_50](https://user-images.githubusercontent.com/44984892/51155085-e827b700-18a8-11e9-9b0d-0dd3d347a9fa.png)
![cos_sim_dis_200](https://user-images.githubusercontent.com/44984892/51155087-e8c04d80-18a8-11e9-97e3-91fe07f0f67f.png)

|มิติ |จำนวนที่ similarity สูงกว่า 0.5 |ความน่าจะเป็น | |
|:-:|--:|:-:|:--|
|2 |166215 / 499500 |33.28% |[-π/3, π/3] / 2π = 1/3 |
|3 |124860 / 499500 |25.00% |1/2 * 2π / 4π sr = 1/4 |
|50 |50 / 499500 |0.01% | |
|200 |0 / 499500 |0.00% | |

เพราะฉะนั้น ในปริภูมิ 200 มิติ สามารถคิดได้ว่า "มี similarity **ตั้ง** 0.5" แทนที่จะคิดว่า "มี similarity **แค่** 0.5" แต่ distribution จริงอาจจะไม่เหมือนกับ Gaussian จึงต้องวิเคราะห์ใหม่ทีหลัง

## 2. Result: similarity of words
<pre>
similar('ปลาดิบ',10)
('อาทิตย์อุทัย', 0.8037128448486328)
('กิมจิ', 0.769778847694397)
('อิเหนา', 0.6771085262298584)
('หมีขาว', 0.6678430438041687)
('จิงโจ้', 0.6606272459030151)
('ซามูไร', 0.6142847537994385)
('มะกะโรนี', 0.5971331596374512)
('กีวี', 0.5338258743286133)
('มังกร', 0.4993358552455902)
('ไก่งวง', 0.49651074409484863)

similar('สวย',10)
('เริด', 0.7605462074279785)
('เซ็กซี่', 0.723678469657898)
('หุ่นดี', 0.6765000820159912)
('ดูดี', 0.6757391691207886)
('เตะตา', 0.6746330261230469)
('อึ๋ม', 0.6711900234222412)
('เป๊ะ', 0.6602514982223511)
('เนี้ยบ', 0.6500211954116821)
('เว่อร์', 0.6493927240371704)
('เริ่ด', 0.6467969417572021)

similar('ผู้ชาย',10)
('ผู้หญิง', 0.914494514465332)
('เด็กผู้ชาย', 0.6376828551292419)
('กะเทย', 0.6262935400009155)
('คนเจ้าชู้', 0.6182562112808228)
('เด็กผู้หญิง', 0.5889356732368469)
('เกย์', 0.5706191658973694)
('เจ้าชู้', 0.5634102821350098)
('ลูกผู้หญิง', 0.5335626602172852)
('ชายหนุ่ม', 0.532731294631958)
('รักเดียว', 0.5316658020019531)
</pre>

|word1 | word2 | similarity |
|:-:|:-:|--:|
|สวย|โรงเรียน| -0.0428314 |
|ไป |อร่อย |0.04088692 |
|ครับ |จุฬา |-0.032523237 |
|วิ่ง |ทาน |0.10898548 |
|การ์ตูน |หนังสือ |0.21807942 |
|สยาม |พารากอน |0.31013966 |

cosine similarity distribution of random two word pair (500000 pairs)<br>
![cos_distribution](https://user-images.githubusercontent.com/44984892/51297774-55c41680-1a54-11e9-80b7-2ac4cad0f73b.png)<br>

|จำนวนที่ similarity สูงกว่า 0.5 | ความน่าจะเป็น || จำนวนที่ similarity สูงกว่า 0.6 | ความน่าจะเป็น ||จำนวนที่ similarity สูงกว่า 0.7 | ความน่าจะเป็น |
|--:|:-:|:-:|--:|:-:|:-:|--:|:-:|
| 10702/5000000| 0.214%|| 6541/5000000| 0.131%|| 0/5000000| 0.000%|

-> approximation as Gaussian จะได้รู้ว่า SD เท่าไร หรือใช้สูตร FWHM = 2√2lnσ<br>

## 3. Result: vector calculation
### 3.1 สมมติฐาน 1: metonymy is a parallel translation
![met_vector](https://user-images.githubusercontent.com/44984892/51070601-7ff18f00-1676-11e9-809e-eda1ae81a817.jpg) <br>
#### 3.1.1 mean metonymization vector 
<a href="https://www.codecogs.com/eqnedit.php?latex=\overrightarrow{f}&space;=&space;\frac{1}{n}&space;\sum_{X}\overrightarrow{XX'}&space;=&space;\frac{1}{n}\left(&space;(\overrightarrow{OA'}&space;-&space;\overrightarrow{OA})&space;&plus;&space;(\overrightarrow{OB'}&space;-&space;\overrightarrow{OB})&space;&plus;&space;\cdots&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\overrightarrow{f}&space;=&space;\frac{1}{n}&space;\sum_{X}\overrightarrow{XX'}&space;=&space;\frac{1}{n}\left(&space;(\overrightarrow{OA'}&space;-&space;\overrightarrow{OA})&space;&plus;&space;(\overrightarrow{OB'}&space;-&space;\overrightarrow{OB})&space;&plus;&space;\cdots&space;\right)" title="\overrightarrow{f} = \frac{1}{n} \sum_{X}\overrightarrow{XX'} = \frac{1}{n}\left( (\overrightarrow{OA'} - \overrightarrow{OA}) + (\overrightarrow{OB'} - \overrightarrow{OB}) + \cdots \right)" /></a> <br>
where X: country, X': metonymy of country

#### 3.1.2 วิธีค้นหา X หรือ X'
1. (country B) + (metonymy A') - (country A) = (metonymy B') : country B + metonymization vector
2. (metonymy B') + (country A) - (metonymy A') = (country B) : metonymy B' - metonymization vector

##### 3.1.2.1 วิธี 1 ค้นหา metonymy
* 'เกาหลี' + ปลาดิบ' - 'ญี่ปุ่น' = ('กิมจิ', 0.7009077072143555) ...
* 'อังกฤษ' + 'ซามูไร' - 'ญี่ปุ่น' = ('ผู้ดี', 0.5545893311500549) ...
* 'เกาหลีเหนือ' + 'ซามูไร' - 'ญี่ปุ่น' = ('โสมแดง', 0.6188918948173523) ...
* 'ญี่ปุ่น' + 'ไก่งวง' - 'ตุรกี' = ('ปลาดิบ', 0.5589873790740967) ...

##### 3.1.2.2 วิธี 2 ค้นหา country
* 'มะกะโรนี' + 'ญี่ปุ่น' - 'ปลาดิบ' = ('อิตาลี', 0.553771436214447) ...
* 'จิงโจ้' + 'เกาหลีใต้' - 'กิมจิ' = ('ออสเตรเลีย', 0.5869286060333252) ...
* 'กีวี' + 'จีน' - 'มังกร' = ('นิวซีแลนด์', 0.5277906060218811) ...
* 'อิเหนา' + 'ญี่ปุ่น' - 'ซามูไร' = ('อินโดนีเซีย', 0.5943317413330078) ...
* 'กระทิง' + 'เกาหลีใต้' - 'กิมจิ' = ('สเปน', 0.4568285346031189) ...
* 'น้ำหอม' + 'เกาหลีใต้' - 'กิมจิ' = ('ฝรั่งเศส', 0.42910343408584595) ...

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
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&0&\cdots&0&b_1\\&space;0&A_{2,2}&\cdots&0&&space;b_2\\&space;\vdots&&\ddots&&\vdots\\&space;0&0&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&0&\cdots&0&b_1\\&space;0&A_{2,2}&\cdots&0&&space;b_2\\&space;\vdots&&\ddots&&\vdots\\&space;0&0&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" title="\begin{pmatrix} x_1'\\x_2'\\\vdots\\x_{200}'\\1 \end{pmatrix} = \begin{pmatrix} A_{1,1}&0&\cdots&0&b_1\\ 0&A_{2,2}&\cdots&0& b_2\\ \vdots&&\ddots&&\vdots\\ 0&0&\cdots&A_{200,200}&b_{200}\\ 0&0&\cdots&0&1 \end{pmatrix} \begin{pmatrix} x_1\\x_2\\\vdots\\x_{200}\\1 \end{pmatrix}" /></a>

#### 3.2.2 Affine Transformation with full matrix
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\&space;A_{2,1}&A_{2,2}&\cdots&A_{2,200}&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\&space;A_{2,1}&A_{2,2}&\cdots&A_{2,200}&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" title="\begin{pmatrix} x_1'\\x_2'\\\vdots\\x_{200}'\\1 \end{pmatrix} = \begin{pmatrix} A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\ A_{2,1}&A_{2,2}&\cdots&A_{2,200}& b_2\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\ 0&0&\cdots&0&1 \end{pmatrix} \begin{pmatrix} x_1\\x_2\\\vdots\\x_{200}\\1 \end{pmatrix}" /></a>

### 3.3 Distance
เพื่อที่จะวิเคราะห์ความสัมพันธ์ระหว่างประเทศกับนามนัย การวัด distance อาจจะมีประโยชน์ แต่ต้องเลือก distance ที่เหมาะสม <br>

#### 3.3.1 Distance between country and metonymy
* Euclidean Distance -> แต่ละมิติต้องเป็น orthogonal basis แต่ word2vec ไม่เหมือนกับ SVD เพราะฉะนั้น ความชัดเจนน้อยลง ถ้าสุ่มเลือกสองจุดในปริภูมิ 200 มิติโดย Gaussian distribution แล้ว distribution ของ Euclidean distance ของสองจุดนี้ก็จะเป็น Gaussian เหมือนกัน โดยมีค่ามัชฌิม 20 ( -> [random_vector.py](https://github.com/nozomiyamada/word2vec/blob/master/random_vector.py) ) `dis_distribution`<br>
![dis_distribution](https://user-images.githubusercontent.com/44984892/51271873-419ffb00-19fb-11e9-9937-313c2e138307.png) <br>

แต่ที่จริง การกระจายของ word vector ไม่เหมือน gaussian <br>
![dis_words](https://user-images.githubusercontent.com/44984892/51267035-60989000-19ef-11e9-90d6-bc6dbe9bd03a.png)
![dis_words_log](https://user-images.githubusercontent.com/44984892/51267032-5ffff980-19ef-11e9-82d8-9ecca5263009.png) <br>

ผลลัพธ์ : Euclidean distance between country and metonymy <br>
<img src="https://user-images.githubusercontent.com/44984892/51169912-aebe6e00-18df-11e9-8873-74bc772b6352.png" width="600px" >

distance ของ metonymization vector ประมาณ 30 
* Wasserstein Embeddings 

#### 3.3.2 Distance among metonymies (or countries)

* ~~Mahalanobis Distance -> เพื่อเปรียบเทียบทั้ง metonymy และสามารถหา prototype ที่มีความนามนัยสูงที่สุดได้ แต่ต้องสมมติการกระจายเป็น Gaussian~~ <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=N(\vec{x}|\vec{\mu},\Sigma)&space;=&space;\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}&space;\exp&space;\left[&space;-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(\vec{x}|\vec{\mu},\Sigma)&space;=&space;\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}&space;\exp&space;\left[&space;-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu})&space;\right]" title="N(\vec{x}|\vec{\mu},\Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} \exp \left[ -\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-\vec{\mu}) \right]" /></a> <br>
ใช้ไม่ได้ เพราะต้องเป็นแบบ **จำนวนข้อมูล > จำนวนมิติ (full rank)** ไม่อย่างนั้น หา inverse matrix ของ covariance matrix ไม่ได้

* k-means : clustering metonymies into subtypes 

* Poincare Embeddings

### 3.4 Canonical Correlation Analysis

## 4. รายชื่อ metonymy ที่พบเจอใน "ไทยรัฐ"
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
|สหรัฐฯ |นกอินทรี, แฮมเบอร์เกอร์ |
|อิตาลี |มะกะโรนี |
|รัสเซีย |หมีขาว |
|ฝรั่งเศส |น้ำหอม |
|เยอรมัน |เบียร์ |
|สเปน |กระทิง |
|โปรตุเกส |ฝอยทอง |
|เนเธอร์แลนด์ |กังหันลม | 

ดูเหมือนว่ามี metonymy สองประเภทใหญ่ ได้แก่ "ของกิน" กับ "สิ่งมีชีวิต"
1. ของกิน : ปลาดิบ กิมจิ มะกะโรนี
2. สิ่งมีชีวิต : ซามูไร มังกร ผู้ดี จิงโจ้ กีวี

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
