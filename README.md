# word2vec : Metonymies of countries

## Method
เก็บข้อมูลมาจาก 'ไทยรัฐ' ทั้งหมด 203624 บทความ (12/01/2562)

text tokenize: PyThaiNLP
gensim.model.word2vec size=200, min_count=5, window=15

วัดความคล้ายคลีงโดยใช้ cos similarity <a href="https://www.codecogs.com/eqnedit.php?latex=\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\cos{\theta}&space;=&space;\frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" title="\cos{\theta} = \frac{\vec{x}\cdot\vec{y}}{|\vec{x}||\vec{y}|}" /></a>
