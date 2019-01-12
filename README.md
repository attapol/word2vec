# word2vec : Metonymies of countries in ไทยรัฐ

## Method
เก็บข้อมูลมาจาก 'ไทยรัฐ' ทั้งหมด 203624 บทความ (12/01/2562) <br>

text tokenize: PyThaiNLP <br>
gensim.model.word2vec size=200, min_count=5, window=15 <br>

วัดความคล้ายคลีงโดยใช้ cos similarity <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" title="\cos{\theta} = \frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" /></a>

## Result: similarity
<pre>
In[19]: similar('ปลาดิบ')
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
</pre>

## Result: vector calculation
![met_vector](https://user-images.githubusercontent.com/44984892/51070601-7ff18f00-1676-11e9-809e-eda1ae81a817.jpg) <br>
### mean metonymization vector 
<a href="https://www.codecogs.com/eqnedit.php?latex=\overrightarrow{f}&space;=&space;\frac{1}{n}&space;\sum_{X}\overrightarrow{XX'}&space;=&space;\frac{1}{n}\left(&space;(\overrightarrow{OA'}&space;-&space;\overrightarrow{OA})&space;&plus;&space;(\overrightarrow{OB'}&space;-&space;\overrightarrow{OB})&space;&plus;&space;\cdots&space;\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\overrightarrow{f}&space;=&space;\frac{1}{n}&space;\sum_{X}\overrightarrow{XX'}&space;=&space;\frac{1}{n}\left(&space;(\overrightarrow{OA'}&space;-&space;\overrightarrow{OA})&space;&plus;&space;(\overrightarrow{OB'}&space;-&space;\overrightarrow{OB})&space;&plus;&space;\cdots&space;\right)" title="\overrightarrow{f} = \frac{1}{n} \sum_{X}\overrightarrow{XX'} = \frac{1}{n}\left( (\overrightarrow{OA'} - \overrightarrow{OA}) + (\overrightarrow{OB'} - \overrightarrow{OB}) + \cdots \right)" /></a> <br>
where X: country, X': metonymy of country

### วิธีค้นหา X หรือ X'
1. (country B) + (metonymy A') - (country A) = (metonymy B') : country B + metonymization vector
2. (metonymy B') + (country A) - (metonymy A') = (country B) : metonymy B' - metonymization vector

#### วิธี 1 ค้นหา metonymy
* 'เกาหลี' + ปลาดิบ' - 'ญี่ปุ่น' = ('กิมจิ', 0.7009077072143555) ...
* 'อังกฤษ' + 'ซามูไร' - 'ญี่ปุ่น' = ('ผู้ดี', 0.5545893311500549) ...

#### วิธี 2 ค้นหา country
* 'มะกะโรนี' + 'ญี่ปุ่น' - 'ปลาดิบ' = ('อิตาลี', 0.553771436214447) ...
* 'จิงโจ้' + 'เกาหลีใต้' - 'กิมจิ' = ('ออสเตรเลีย', 0.5869286060333252) ...
* 'กีวี' + 'จีน' - 'มังกร' = ('นิวซีแลนด์', 0.5277906060218811) ...
