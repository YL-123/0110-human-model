from Humanagender import Agender


def demo():
    imgname='./test1.png'
    cfg = Agender()
    age, gender = cfg.predict(imgname)    
    print gender, age

if __name__ == '__main__':
	demo()
