# word2vec : Metonymies of countries in ไทยรัฐ

## 1. Method
หา correspondence ระหว่างประเทศกับนามนัย (metonymy) โดยใช้ deep learning (word2vec) <br>
เก็บข้อมูลมาจาก "ไทยรัฐ" ทั้งหมด 203624 บทความ (12/01/2562) <br>
ใช้ CBOW model (ฝึกโดยเนื้อหาบทความเท่านั้น)

### 1.1 python toolkit
tokenizer: PyThaiNLP <br>
gensim.model.word2vec (size=200, min_count=5, window=15) <br>

วัดความคล้ายคลีงโดยใช้ cos similarity <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" title="\cos{\theta} = \frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" /></a>

## 2. Result: similarity
<pre>
In [19]: similar('ปลาดิบ')
('กิมจิ', 0.7630575299263)
('จิงโจ้', 0.6887362599372864)
('อิเหนา', 0.6832948923110962)
('มะกะโรนี', 0.6281756162643433)
('ซามูไร', 0.5945858955383301)
('มักกะโรนี', 0.5800601243972778)
('ฝอยทอง ', 0.492371529340744)
('ไก่งวง', 0.4914326071739197)
('กีวี', 0.49096113443374634)
('ตะลุย', 0.47837239503860474)

In [78]: similar('สวย')
('เริด', 0.7534685134887695)
('เตะตา', 0.7298843860626221)
('เซ็กซี่', 0.7285832166671753)
('หุ่นดี', 0.7116822600364685)
('อึ๋ม', 0.6884818077087402)
('เว่อร์', 0.6793030500411987)
('ดูดี', 0.6685600280761719)
('เป๊ะ', 0.663704514503479)
('เนี้ยบ', 0.6614724397659302)
('เปล่งประกาย', 0.6481001973152161)

In [118]: similar('ผู้ชาย')
('ผู้หญิง', 0.9142853021621704)
('กะเทย', 0.6702405214309692)
('เด็กผู้ชาย', 0.6212347149848938)
('คนเจ้าชู้', 0.6135973930358887)
('เจ้าชู้', 0.5497159957885742)
('เค้า', 0.5450853109359741)
('เด็กผู้หญิง', 0.5440890789031982)
('ชายหนุ่ม', 0.5399799346923828)
('โรคจิต', 0.5319931507110596)
('เกย์', 0.5251267552375793)
</pre>

## 3. Result: vector calculation
### 3.1 สมมติฐาน 1: metonymy is a parallel shift 
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

### 3.2 สมมติฐาน 2: metonymy is Affine Transformation
<a href="https://www.codecogs.com/eqnedit.php?latex=\vec{x'}&space;=&space;A\vec{x}&space;&plus;&space;\vec{b}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\vec{x'}&space;=&space;A\vec{x}&space;&plus;&space;\vec{b}" title="\vec{x'} = A\vec{x} + \vec{b}" /></a> <br>
ถ้าสมมติ Affine Transformation ต้องหา linear transformation matrix A กับ shift vector b <br>
ในกรณีนี้ vector มี 200 มิติ สัมประสิทธิ์ก็มีทั้งหมด 200^2 (A) + 200 (b) = 40200 ตัว (ต้องการ 201 vector ที่ต่างกัน) แต่เก็บตัวอย่างมากขนาดนี้ไม่ได้ เพราะฉะนั้นต้องการ regression model อะไรสักอย่าง เช่น least square

### 3.3 Distance
เพื่อที่จะวิเคราะห์ความสัมพันธ์ระหว่างประเทศกับนามนัย การวัด distance อาจจะมีประโยชน์ แต่ต้องเลือก distance ที่เหมาะสม <br>
* Euclidean Distance -> ต้องเป็น orthogonal basis ใช้ได้หรือเปล่า?
* Mahalanobis Distance -> เพื่อเปลียบเทียบทั้ง metonymy และหา prototype ได้

## 4. metonymy ที่พบเจอใน "ไทยรัฐ"
|ประเทศ  |นามนัย  |
|:-:|:--|
|ญี่ปุ่น  |ปลาดิบ, ซามูไร  |
|จีน  |มังกร, แผ่นดินใหญ่  |
|เกาหลี |กิมจิ, โสมแดง(เหนือ) |
|อังกฤษ |ผู้ดี |
|ตุรกี |ไก่งวง |
|ออสเตรเลีย |จิงโจ้ |
|นิวซีแลนด์ |กีวี |
|อินโดนีเซีย |อิเหนา |
|อิตาลี |มะกะโรนี |

ดูเหมือนว่ามี metonymy สองประเภทใหญ่ ได้แก่ "ของกิน" กับ "สิ่งมีชีวิต"
1. ของกิน : ปลาดิบ กิมจิ มะกะโรนี
2. สิ่งมีชีวิต : ซามูไร มังกร ผู้ดี จิงโจ้ กีวี

## 5. Problems
* คำศัพท์บางคำหาไม่เจอโดยใช้วิธีนี้ เช่น คำว่า 'ช้าง' ซึ่งเป็นนามนัยของประเทศไทย อาจจะเป็นเพราะคำว่าข้างใช้ในบริบทธรรมดาด้วย ส่วนคำว่า 'ปลาดิบ' ไม่ค่อยปรากฏในความหมายดั้งเดิม (ปรากฏเฉพาะในกรณีพูดถึงประเทศญี่ปุ่น)
* พูดถึง metonymy แล้ว ควรวิเคราะห์เชิง coginitive linguistics แต่ไม่รู้ว่าจะเชื่อมโยงกันได้อย่างไง ยกเว้นแนวคิด prototype
* 

## 6. Extra
* 'จีน' + 'โตเกียว' - 'ญี่ปุ่น' = ('ปักกิ่ง', 0.7812713384628296) ...
* 'จีน' + 'เกียวโต' - 'ญี่ปุ่น' = ('ฉงชิ่ง', 0.6020291447639465) ...
* 'ต้มยำกุ้ง' + 'ญี่ปุ่น' - 'ไทย' = ('แซลมอน', 0.4649949371814728) ...
* 'ฮ่องเต้' + 'ญี่ปุ่น' - 'จีน' = ('จักรพรรดิ', 0.5973250865936279) ...
* 'ภรรยา' - 'สาว' = ('บุตรชาย', 0.49089011549949646) ...
* 'ประวิตร' - 'นาฬิกา' = ('ประยุทธ์', 0.5291631817817688) ...
