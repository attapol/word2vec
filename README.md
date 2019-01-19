# word2vec : Metonymies of countries in ไทยรัฐ

## 1. Method
หา correspondence ระหว่างประเทศกับนามนัย (metonymy) โดยใช้ deep learning (word2vec) <br>
เก็บข้อมูลมาจาก "ไทยรัฐ" ทั้งหมด 362985 บทความ (18/01/2562) <br>
ใช้ ทั้ง skip-gram model และ CBOW แล้วเปรียบเทียบกัน (ฝึกโดยเนื้อหาบทความเท่านั้น)

### 1.1 python toolkit
tokenizer: `pythainlp.tokenize.word_tokenize` <br>
word2vec: `gensim.model.word2vec` (sg=1, size=200, min_count=5, window=15) <br>

### 1.2 cosine similarity 
วัดความคล้ายคลึงโดยใช้ cosine similarity <br>
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

## 2. Result: similarity of words
<pre>
met.similar('ปลาดิบ')
('อาทิตย์อุทัย', 0.7142832279205322)
('กิมจิ', 0.6960911750793457)
('ซามูไร', 0.6283251047134399)
('ญี่ปุ่น', 0.6202974915504456)
('ประเทศญี่ปุ่น', 0.609769880771637)
('โอกะ', 0.5708627104759216)
('Tinydoll', 0.5689390301704407)
('โตะ"', 0.5666331052780151)
('เกะ', 0.562018096446991)
('ดะ', 0.5590271353721619)
('สึ', 0.5478522777557373)
('มัตสึ', 0.5467077493667603)
('ซุ', 0.5442054867744446)
('โตะ', 0.5436292290687561)
('กิ"', 0.5390536189079285)
('จิงโจ้', 0.5358957052230835)
('ชาวญี่ปุ่น', 0.5323625206947327)
('โสมขาว', 0.5323294997215271)
('คัสซึโตะ', 0.5303484201431274)
('ลอดช่อง', 0.5275979042053223)

met.similar('สวย',10)
('เริด', 0.7596404552459717)
('เซ็กซี่', 0.7574315071105957)
('เว่อร์', 0.7437990307807922)
('อึ๋ม', 0.7304662466049194)
('หุ่นดี', 0.7293158769607544)
('เป็นสาว', 0.6788187026977539)
('ชวนมอง', 0.6742486953735352)
('ดูดี', 0.6729604005813599)
('ลุค', 0.671097993850708)

met.similar('ผู้ชาย',10)
('ผู้หญิง', 0.9339771270751953)
('คนเจ้าชู้', 0.6667338013648987)
('เจ้าชู้', 0.6652568578720093)
('คบ', 0.6608400344848633)
('เด็กผู้ชาย', 0.6581286191940308)
('คนอื่น', 0.6470351219177246)
('กะเทย', 0.6453133225440979)
('จีบ', 0.6446007490158081)
('เค้า', 0.6411833167076111)
('ชายหนุ่ม', 0.6373834609985352)
</pre>

cosine similarity distribution of random two word pair (500000 pairs)<br>
![sim_distribution](https://user-images.githubusercontent.com/44984892/51410970-b2304e80-1b98-11e9-9fca-0d688584972d.png) <br>
Gaussian fitting : mean = 0.1917, SD = 0.1257 <br>
mean ไม่ใช่ 0 แสดงว่า vector เหล่านี้เป็น uneven distribution ในปริภูมิ 200 มิติ

|x |0.50 |0.52|0.54|0.56|0.58|0.60|0.62|0.64|0.66|0.68|0.70|
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|ความน่าจะเป็นที่สูงกว่า x [%]| 2.186 |1.733| 1.362 |1.057| 0.819 | 0.631|0.480 |0.357| 0.264 | 0.188| 0.131|


note : ถ้าใช้ CBOW model ค่ามัฌขิมจะเป็นประมาณ 0

|word1 | word2 | similarity | distance |
|:-:|:-:|--:|--:|
|สวย|โรงเรียน|0.1545247 |3.7336485 |
|ไป |อร่อย |0.17561817 |3.8372965 |
|ครับ |จุฬา |0.07391273 |4.64459 |
|การ์ตูน |หนังสือ |0.34722275 |3.8463798 |
|สยาม |พารากอน |0.374858 |4.1474586 |

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
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&0&\cdots&0&b_1\\&space;0&A_{2,2}&\cdots&0&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;0&0&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&0&\cdots&0&b_1\\&space;0&A_{2,2}&\cdots&0&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;0&0&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" title="\begin{pmatrix} x_1'\\x_2'\\\vdots\\x_{200}'\\1 \end{pmatrix} = \begin{pmatrix} A_{1,1}&0&\cdots&0&b_1\\ 0&A_{2,2}&\cdots&0& b_2\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ 0&0&\cdots&A_{200,200}&b_{200}\\ 0&0&\cdots&0&1 \end{pmatrix} \begin{pmatrix} x_1\\x_2\\\vdots\\x_{200}\\1 \end{pmatrix}" /></a><br>
ในกรณีนี้ คำนวณแต่ละสัมประสิทธิ์ได้โดย simple linear regression <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=x_i'=A_{i,i}x_i&plus;b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i'=A_{i,i}x_i&plus;b_i" title="x_i'=A_{i,i}x_i+b_i" /></a>

Result: detA (product of eigenvalue A<sub>i,i</sub>) = -4.824009094491605e-46 <br>
cosine similarity of "mean metonymization vector" and "parallel translation b" = 0.8057
เพราะฉะนั้น ตีความได้ว่า ถ้าใช้สมมติฐานนี้ matrix A เกือบไม่ส่งผล และ metonymization จะเกิดจาก parallel translation b เป็นหลัก (แต่อาจจะมีมิติที่ส่งผลมากกว่ามิติอื่นก็ได้ ต้องวิเคราะห์สัมประสิทธิ์ของ A)

#### 3.2.2 Affine Transformation with full matrix
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\&space;A_{2,1}&A_{2,2}&\cdots&A_{2,200}&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{pmatrix}&space;x_1'\\x_2'\\\vdots\\x_{200}'\\1&space;\end{pmatrix}&space;=&space;\begin{pmatrix}&space;A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\&space;A_{2,1}&A_{2,2}&\cdots&A_{2,200}&&space;b_2\\&space;\vdots&\vdots&\ddots&\vdots&\vdots\\&space;A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\&space;0&0&\cdots&0&1&space;\end{pmatrix}&space;\begin{pmatrix}&space;x_1\\x_2\\\vdots\\x_{200}\\1&space;\end{pmatrix}" title="\begin{pmatrix} x_1'\\x_2'\\\vdots\\x_{200}'\\1 \end{pmatrix} = \begin{pmatrix} A_{1,1}&A_{1,2}&\cdots&A_{1,200}&b_1\\ A_{2,1}&A_{2,2}&\cdots&A_{2,200}& b_2\\ \vdots&\vdots&\ddots&\vdots&\vdots\\ A_{200,1}&A_{200,2}&\cdots&A_{200,200}&b_{200}\\ 0&0&\cdots&0&1 \end{pmatrix} \begin{pmatrix} x_1\\x_2\\\vdots\\x_{200}\\1 \end{pmatrix}" /></a><br>
ในกรณีนี้ คำนวณแต่ละสัมประสิทธิ์ได้โดย multiple linear regression <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=x_i'=A_{i,1}x_1&plus;A_{i,2}x_2&plus;\cdots&plus;A_{i,200}x_{200}&plus;b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i'=A_{i,1}x_1&plus;A_{i,2}x_2&plus;\cdots&plus;A_{i,200}x_{200}&plus;b_i" title="x_i'=A_{i,1}x_1+A_{i,2}x_2+\cdots+A_{i,200}x_{200}+b_i" /></a>

Result: detA ~ 0 <br>
cosine similarity of "mean metonymization vector" and "parallel translation b" = 0.2449
อิทธิพลจาก parallel translation b ยังใหญ่ แต่น้อยลงนิดหน่อย เพราะฉะนั้น ถ้าใช้ full matrix A อาจจะค้นหา metonyms ได้ดีกว่า <br>

เช่น ในข้อมูลที่ใช้เพื่อ regression ไม่มีคำว่า 'กังหันลม' ซึ่งเป็น metonym ของ 'เนเธอร์แลนด์'
แต่ 'เนเธอร์แลนด์' + ปลาดิบ' - 'ญี่ปุ่น' = 'หมีขาว', 'จิงโจ้', 'อาทิตย์อุทัย' ... โดย 'กังหันลม' ไม่ปรากฎ ส่วน ใช้ Affine Transformation แล้ว ได้พบเจอ ('กังหันลม', 0.4553275909584193) เพราะฉะนั้น เอาสองวิธี (parallel taranslation & Affine transformation) มารวามกันดีกว่า

### 3.3 Distance
เพื่อที่จะวิเคราะห์ความสัมพันธ์ระหว่างประเทศกับนามนัย การวัด distance อาจจะมีประโยชน์ แต่ต้องเลือก distance ที่เหมาะสม <br>

#### 3.3.1 Distance between country and metonymy
* Euclidean Distance -> แต่ละมิติต้องเป็น orthogonal basis แต่ word2vec ไม่เหมือนกับ SVD เพราะฉะนั้น ความชัดเจนน้อยลง ถ้าสุ่มเลือกสองจุดในปริภูมิ 200 มิติโดย Gaussian distribution แล้ว distribution ของ Euclidean distance ของสองจุดนี้ก็จะเป็น Gaussian เหมือนกัน โดยมีค่ามัชฌิม 20 ( -> [random_vector.py](https://github.com/nozomiyamada/word2vec/blob/master/random_vector.py) ) `dis_distribution`<br>
![dis_distribution_random](https://user-images.githubusercontent.com/44984892/51271873-419ffb00-19fb-11e9-9937-313c2e138307.png) <br>

แต่ที่จริง การกระจายของ word vector ไม่เหมือน gaussian <br>
![dis_distribution](https://user-images.githubusercontent.com/44984892/51412364-e9edc500-1b9d-11e9-86fe-82396f84ec9c.png)

ผลลัพธ์ : Euclidean distance between country and metonymy <br>
<img src="https://user-images.githubusercontent.com/44984892/51412426-202b4480-1b9e-11e9-98de-154b4b1435c2.png" width="600px" >

distance ของ metonymization vector : mean 3.2456
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
