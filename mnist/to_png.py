#　MNISTデータを読み込み、PNGファイルに変換して保存する。
#
#   size:
#       出力する画像の枚数。0以下なら強制的に1枚出力。
#
#   output_path:
#       画像の出力先のパス。フォルダの区切りは\\なのに注意
#
#   save_test: テスト用データから抽出する
#   save_train: トレーニング用データから抽出する
#

from PIL import Image
import mnist.getdata as mt

def to_png(x,name,w=28,h=28):
    img = x
    b = img.reshape(w,h)
    outImg = Image.fromarray(b)
    outImg = outImg.convert("RGB")
    outImg.save(name + ".png")
    
    
def save_train(size,output_path):

    train_size = 60000
    if(size <= 0):
        size = 1
        
    if(size >= train_size):
        (xx,xt) = mt.get_traindata()
    else:
        (xx,xt) = mt.get_traindata_choiced(size)
        
    for i in range(len(xx)):
        # PNGファイル変換
        to_png(xx[i],output_path + str(i+1) + "_lv_" + str(xt[i]) + "_train.png",28,28)

        #img = xx[i]
        #b = img.reshape((28,28))
        #outImg = Image.fromarray(b)
        #outImg = outImg.convert("RGB")
        #outImg.save(output_path + str(i+1) + "_lv_" + str(xt[i]) + "_train.png")


def save_test(size,output_path):

    test_size = 10000
    if(size <= 0):
        size = 1
        
    if(size >= test_size):
        (xx,xt) = mt.get_testdata()
    else:
        (xx,xt) = mt.get_testdata_choiced(size)
        
    for i in range(len(xx)):
        # PNGファイル変換
        to_png(xx[i],output_path + str(i+1) + "_lv_" + str(xt[i]) + "_test.png",28,28)
        
        #img = xx[i]
        #b = img.reshape((28,28))
        #outImg = Image.fromarray(b)
        #outImg = outImg.convert("RGB")
        #outImg.save(output_path + str(i+1) + "_lv_" + str(xt[i]) + "_test.png")




