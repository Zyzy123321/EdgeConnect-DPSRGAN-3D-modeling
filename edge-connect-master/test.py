import os

from main import main
import xlrd2
import xlwt
from PIL import Image

def writepic(resultpic,resultexcel):
    img = Image.open(resultpic)
    img = img.convert('RGBA')
    img1=img.load()
    y,x=img.size[0],img.size[1]
    print(x,y)


    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    Rsheet = book.add_sheet('R', cell_overwrite_ok=True)
    Gsheet = book.add_sheet('G', cell_overwrite_ok=True)
    Bsheet = book.add_sheet('B', cell_overwrite_ok=True)
    # Hsheet = book.add_sheet('H', cell_overwrite_ok=True)
    # Ssheet = book.add_sheet('S', cell_overwrite_ok=True)
    # Vsheet = book.add_sheet('V', cell_overwrite_ok=True)

    for i in range(0, y):
        Rsheet.write(0, i, i)
        Gsheet.write(0, i, i)
        Bsheet.write(0, i, i)
        # Hsheet.write(0, i, i)
        # Ssheet.write(0, i, i)
        # Vsheet.write(0, i, i)

    for i in range(0,x):
        for j in range(0,y):
            r = img1[j, i][0]
            g = img1[j, i][1]
            b = img1[j, i][2]
            Rsheet.write(i+1, j, r)
            Gsheet.write(i+1, j, g)
            Bsheet.write(i+1, j, b)

            # mx=max(r,g,b)
            # mn=min(r,g,b)
            # df=mx-mn
            # if mx==mn:
            #     h=0
            # elif mx==r:
            #     h = (60 * ((g - b) / df) + 360) % 360
            # elif mx == g:
            #     h = (60 * ((b - r) / df) + 120) % 360
            # elif mx == b:
            #     h = (60 * ((r - g) / df) + 240) % 360
            # if mx == 0:
            #     s = 0
            # else:
            #     s = df / mx
            # v = mx
            # Hsheet.write(y-i,j,h)
            # Ssheet.write(y-i,j,s)
            # Vsheet.write(y-i,j,v)

    book.save(resultexcel)

def createpicture(filepath,picsavepath,masksavepath):

    wb=xlrd2.open_workbook(filepath)
    rsheet=wb.sheet_by_name("R")
    gsheet=wb.sheet_by_name("G")
    bsheet=wb.sheet_by_name("B")
    x = rsheet.nrows
    y = rsheet.ncols
    print(rsheet.cell(23,0).value)

    im = Image.new("RGB", (y, x-1))   #创建图片
    mask=Image.new("RGB", (y, x-1))
    #将rgb转化为像素

    for i in range(0,y):
        for j in range(0,x-1):
            r =int(rsheet.cell(j + 1, i).value)
            g =int(gsheet.cell(j + 1, i).value)
            b =int(bsheet.cell(j + 1, i).value)
            im.putpixel((i,j),(r,g,b))

            if r!=255 or g!=255 or b!=255:
                mask.putpixel((i,j),(0,0,0))
            else:
                mask.putpixel((i, j), (255, 255,255))

    # images=np.array(im)
    # print(im.size,images)
    #im.show()
    im.save(picsavepath)
    mask.save(masksavepath)


def get_files():
    for path,dirnames,filenames in os.walk(r'F:/论文/训练数据/原始钻孔图像1excel'):
        for filename in filenames:
            filepath=os.path.join(path,filename)
            #print(filepath)
            #filepath=r'F:/论文/训练数据/原始钻孔图像excel/14.8.xlsx'
            name = filepath.split("\\")[-1]
            if len(name.split("."))==2:
                picname=name.split(".")[0]
            else:
                picname = name.split(".")[0]+'.'+name.split(".")[1]
            picsavepath=r'F:/论文/训练数据/原始钻孔图像1/image/'+picname+'.png'
            masksavepath=r'F:/论文/训练数据/原始钻孔图像1/mask/'+picname+'.png'
            resultpic=r'F:/论文/训练数据/结果图/image/'+picname+'.png'
            resultexcel=r'F:/论文/训练数据/结果excel/%s.xls'%picname

            createpicture(filepath,picsavepath,masksavepath)

main(mode=1)

#writepic(picsavepath,resultexcel)
#get_files()