# word2vec : Metonyms of countries in ไทยรัฐ

## 1. Method
หา correspondence ระหว่างประเทศกับนามนัย (metonymy) โดยใช้ deep learning (word2vec) <br>
เก็บข้อมูลมาจาก "ไทยรัฐ" ทั้งหมด 362985 บทความ (18/01/2562) <br>
ใช้ ทั้ง CBOW และ skip-gram model แล้วเปรียบเทียบกัน (ฝึกโดยเนื้อหาบทความเท่านั้น)

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

## 2. Result: similarity & distance of word pair
<table>
<tr>
  <th>CBOW</th>
  <th>skip-gram</th>
</tr>
<tr>
  <td>
    <pre>
met.similar('ปลาดิบ')
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
('ลอดช่อง', 0.4712417721748352)
('ฝอยทอง ', 0.4534761905670166)
('ชุบตัว', 0.4485313594341278)
('ตะลุย', 0.43987876176834106)
('ผู้ดี', 0.4398248493671417)
('ผีดิบ', 0.43067649006843567)
('น้ำหอม', 0.42870694398880005)
('เรกเก้', 0.40906381607055664)
('"เบรน', 0.39751094579696655)
('ขุนพล', 0.3972066640853882)
    </pre>
  </td>
  <td>
    <pre>
met2.similar('ปลาดิบ')
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
    </pre>
  </td>
</tr>
<tr>
  <td>
    <pre>
met.similar('มังกร')
('จิงโจ้', 0.5348995327949524)
('กิมจิ', 0.5091280341148376)
('อาทิตย์อุทัย', 0.5051579475402832)
('ปลาดิบ', 0.4993358254432678)
('หมีขาว', 0.46833932399749756)
('ราชสีห์', 0.4622499346733093)
('ค้างคาว', 0.451012521982193)
('อิเหนา', 0.4478691816329956)
('มะกะโรนี', 0.44519323110580444)
('ฝอยทอง ', 0.4337674379348755)
('ซามูไร', 0.4204603433609009)
('อัคคี', 0.4122592806816101)
('อสูร', 0.4122066795825958)
('"อี้', 0.3966093957424164)
('ระเริง', 0.3946952819824219)
('หยก', 0.39437443017959595)
('เหลือง', 0.38968992233276367)
('ตะลุย', 0.3896721303462982)
('เหยี่ยว', 0.3853123188018799)
('ปลากัด', 0.38338610529899597)
    </pre>
  </td>
  <td>
    <pre>
met2.similar('มังกร')
('หงส์', 0.5328996777534485)
('หยก', 0.5165324807167053)
('สิงโต', 0.5100626945495605)
('วานร', 0.5092955231666565)
('กว่าง', 0.49471861124038696)
('ส.อู๊ดดี้', 0.4928019642829895)
('ฟ้า', 0.4891420006752014)
('โปลิศ', 0.4846121668815613)
('ฉวน', 0.4791263937950134)
('ราชสีห์', 0.4759259521961212)
('โป้', 0.4735491871833801)
('ขุนพล', 0.4707939028739929)
('กังฟู', 0.4706372618675232)
('"อี้', 0.4696926474571228)
('“อี้', 0.4677185118198395)
('อิทธิฤทธิ์', 0.46736180782318115)
('พญา', 0.46702125668525696)
('พิฆาต', 0.4620720148086548)
('ใจเพชร', 0.46157917380332947)
('สิงห์', 0.4611966907978058)
    </pre>
  </td>
</tr>
<tr>
  <td>
    <pre>
 met.similar('ผู้ชาย',10)
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
  </td>
  <td>
    <pre>
met2.similar('ผู้ชาย',10)
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
|สวย|โรงเรียน|-0.0428314| 0.1545247 |53.877216 | 3.7336485 |
|ไป |อร่อย |0.04088692|0.17561817 |50.424267|3.8372965 |
|ครับ |จุฬา |-0.032523237 | 0.07391273 |52.455723 | 4.64459 |
|การ์ตูน |หนังสือ |0.21807942 | 0.34722275 | 45.07515 | 3.8463798 |
|สยาม |พารากอน |0.31013966 | 0.374858 | 34.87113 | 4.1474586 |

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
| 'เกาหลีใต้' + ปลาดิบ' - 'ญี่ปุ่น' | ('อาทิตย์อุทัย', 0.6295)<br>('อิเหนา', 0.6137)<br>('กิมจิ', 0.6110) ... | ('โสมขาว', 0.7123)<br>('กิมจิ', 0.6929)<br>('โสมแดง', 0.5844) ... |
| 'อังกฤษ' + 'ซามูไร' - 'ญี่ปุ่น' | ('ผู้ดี', 0.5536)<br>('หมีขาว', 0.5016)<br>('เวลส์', 0.4929) ... | ('ผู้ดี', 0.6484)<br>('พรีเมียร์', 0.6449)<br>('คำราม', 0.6180) ... |
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

|country : metonym | mean metonymization vector | Affine Transformation |
|:-:|:-:|:-:|
|ญี่ปุ่น : ปลาดิบ | 0.814376260782165 | 0.758601779183857 |
|ญี่ปุ่น : ซามูไร | 0.7398626506470664 | 0.823747168402788 |
|ญี่ปุ่น : นินจา | 0.4485568436522527 |0.7408583027819111|
|ญี่ปุ่น : อาทิตย์อุทัย |0.7926451916486672 | 0.7973922129586257 |
|จีน : มังกร |0.6448982167504311 | 0.9996140535040609 |
|เกาหลี : กิมจิ |0.7844103622475953 | 0.9991522213997927 |
|เกาหลีเหนือ : โสมแดง |0.8672777197761677 | 0.9993399396725856 |
|เกาหลีใต้ : โสมขาว |0.8487338701615814 | 0.9992611952491016 |
|อังกฤษ : ผู้ดี |0.8062117758554058 | 0.9984084064099905 |
|ตุรกี : ไก่งวง |0.6188068272787861 | 0.999741675968432 |
|ออสเตรเลีย : จิงโจ้ |0.7708969496916427 | 0.9996214946014829 |

ในกรณี 'ญี่ปุ่น' ข้อมูลที่ใช้ไม่ใช่ one-to-one mapping เพราะฉะนั้น similarity ก็ประมาณ 0.7 - 0.8 ส่วน metonymy ที่เป็น one-to-one mapping ได้ similarity ที่มากกว่า 0.99 แสดงว่า regression model นี้พอสมควร

### 3.3 Distance
เพื่อที่จะวิเคราะห์ความสัมพันธ์ระหว่างประเทศกับนามนัย การวัด distance อาจจะมีประโยชน์ แต่ต้องเลือก distance ที่เหมาะสม <br>

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
