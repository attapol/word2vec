# word2vec : Metonymies of countries

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
'ปลาดิบ' - 'ญี่ปุ่น' + 'เกาหลี' = ('กิมจิ', 0.7009077072143555) <br>
'มะกะโรนี' - 'ปลาดิบ' + 'ญี่ปุ่น' = ('อิตาลี', 0.553771436214447) <br>
![met_vector](https://user-images.githubusercontent.com/44984892/51070601-7ff18f00-1676-11e9-809e-eda1ae81a817.jpg) <br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\overrightarrow{f}&space;=&space;\overrightarrow{OA'}&space;-&space;\overrightarrow{OA}&space;=&space;\overrightarrow{OB'}&space;-&space;\overrightarrow{OB}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\overrightarrow{f}&space;=&space;\overrightarrow{OA'}&space;-&space;\overrightarrow{OA}&space;=&space;\overrightarrow{OB'}&space;-&space;\overrightarrow{OB}" title="\overrightarrow{f} = \overrightarrow{OA'} - \overrightarrow{OA} = \overrightarrow{OB'} - \overrightarrow{OB}" /></a>

### วิธีคำนวณ
1. metonymy B' - metonymy A' + country A = country B
2. metonymy A' - country A + country B = metonymy B'
