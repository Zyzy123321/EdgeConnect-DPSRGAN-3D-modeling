import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def convolve(filter,mat,padding,strides):
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:,:,i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0,mat_size[0],strides[1]):
                    temp.append([])
                    for k in range(0,mat_size[1],strides[0]):
                        val = (filter*pad_mat[j:j+filter_size[0],k:k+filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j:j + filter_size[0],k:k + filter_size[1]]).sum()
                    channel[-1].append(val)

            result = np.array(channel)

    return result

def linear_convolve(filter,mat,padding=None,strides=[1,1]):

    result = None
    filter_size = filter.shape
    if len(filter_size) == 2 and 1 in filter_size:
        if padding == None or len(padding) < 2:
            if filter_size[1] == 1:
                padding = [filter_size[0]//2,filter_size[0]//2]
            elif filter_size[0] == 1:
                padding = [filter_size[1]//2,filter_size[1]//2]
        if filter_size[0] == 1:
            result = convolve(filter,mat,[0,0,padding[0],padding[1]],strides)
        elif filter_size[1] == 1:
            result = convolve(filter, mat, [padding[0],padding[1],0,0], strides)

    return result

def _2_dim_divided_convolve(filter,mat):

    result = None
    if 1 in filter.shape:
        result = linear_convolve(filter,mat)
        result = linear_convolve(filter.T,result)

    return result

def judgeConnect(m2,threshold):
    e = 0.01
    s = []
    cood = []
    for i in range(m2.shape[0]):
        cood.append([])
        for j in range(m2.shape[1]):
            cood[-1].append([i,j])
            if abs(m2[i,j] - 255) < e:
                s.append([i,j])
    cood = np.array(cood)

    while not len(s) == 0:
        index = s.pop()
        jud = m2[max(0, index[0] - 1):min(index[0] + 2, m2.shape[1]), max(0, index[1] - 1):min(index[1] + 2, m2.shape[0])]
        jud_i = cood[max(0, index[0] - 1):min(index[0] + 2, cood.shape[1]), max(0, index[1] - 1):min(index[1] + 2, cood.shape[0])]
        jud = (jud > threshold[0])&(jud < threshold[1])
        jud_i = jud_i[jud]
        for i in range(jud_i.shape[0]):
            s.append(list(jud_i[i]))
            m2[jud_i[i][0],jud_i[i][1]] = 255

    return m2


def DecideAndConnectEdge(g_l,g_t,threshold = None):
    if threshold == None:
        lower_boundary = g_l.mean()*0.5
        #print(lower_boundary)
        threshold = [lower_boundary,lower_boundary*2]

    result = np.zeros(g_l.shape)

    for i in range(g_l.shape[0]):
        for j in range(g_l.shape[1]):
            isLocalExtreme = True
            eight_neiborhood = g_l[max(0,i-1):min(i+2,g_l.shape[0]),max(0,j-1):min(j+2,g_l.shape[1])]
            if eight_neiborhood.shape == (3,3):
                if g_t[i,j] <= -1:
                    x = 1/g_t[i,j]
                    first = eight_neiborhood[0,1] + (eight_neiborhood[0,1] - eight_neiborhood[0,0])*x
                    x = -x
                    second = eight_neiborhood[2,1] + (eight_neiborhood[2,2] - eight_neiborhood[2,1])*x
                    if not (g_l[i,j] > first and g_l[i,j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] >= 1:
                    x = 1 / g_t[i, j]
                    first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 2] - eight_neiborhood[0, 1]) * x
                    x = -x
                    second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 1] - eight_neiborhood[2, 0]) * x
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] >= 0 and g_t[i,j] < 1:
                    y = g_t[i, j]
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[0, 2] - eight_neiborhood[1, 2]) * y
                    y = -y
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[1, 0] - eight_neiborhood[2, 0]) * y
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] < 0 and g_t[i,j] > -1:
                    y = g_t[i, j]
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[1, 2] - eight_neiborhood[2, 2]) * y
                    y = -y
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[0, 0] - eight_neiborhood[1, 0]) * y
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
            if isLocalExtreme:
                result[i,j] = g_l[i,j]       #非极大值抑制
    #print(np.array(result).flatten())
    threshold[1],threshold[0]=Otsu(result)
    result[result>=threshold[1]] = 255
    result[result<=threshold[0]] = 0


    result = judgeConnect(result,threshold)
    result[result!=255] = 0
    return result

def Otsu(img):
    n=np.array(img).flatten()
    n.sort()
    m=n[n !=0]
    k=[]
    j=[]
    l=[]
    sigama=[]

    maxvalue=(m[-1]//10).astype(int)
    num=m.shape[0]

    for i in range(0,maxvalue+1):
        k.append([])
    for i in m:
        z=(i//10).astype(int)
        k[z].append(i)

    for i in range(0,maxvalue+1):
        l.append(len(k[i]))
        if l[i]==0:
            j.append(0)
        else:
            j.append(sum(k[i]))

    et=[]
    eti=[]
    ei=[]
    pt=[]
    pti=[]
    pi=[]

    for i in range(1,maxvalue+2):
        t=i//2
        a=0
        b=0
        if i==1:
            et.append(0)
            eti.append(t*10+5)
            for q in range(t+1,maxvalue+1):
                a+=l[q]*(q*10+5)
                b+=l[q]
            ei.append(a/b)
            pt.append(0)
            pti.append(l[0]/num)
            pi.append(b/num)
        elif 1<i<maxvalue+1:
            a=0
            b=0
            for q in range(t):
                a+=l[q]*(q*10+5)
                b+=l[q]
            et.append(a/b)
            pt.append(b/num)
            a=0
            b=0
            for x in range(t,i):
                a+=l[x]*(x*10+5)
                b+=l[x]
            eti.append(a/b)
            pti.append(b/num)
            a=0
            b=0
            for y in range(i,maxvalue+1):
                a+=l[y]*(y*10+5)
                b+=l[y]
            ei.append(a/b)
            pi.append(b/num)
        else:
            a=0
            b=0
            for q in range(t):
                a+=l[q]*(q*10+5)
                b+=l[q]
            et.append(a/b)
            pt.append(b/num)
            a = 0
            b = 0
            for x in range(t, i):
                a += l[x]*(x*10+5)
                b += l[x]
            eti.append(a / b)
            pti.append(b/num)
            ei.append(0)
            pi.append(0)

    e=0
    for i in range(maxvalue+1):
        e+=((i*10+5)*l[i]/num)

    ut=[]
    uti=[]
    ui=[]
    for i in range(1,maxvalue+2):
        t=i//2
        c=0
        if i==1:
            ut.append(0)
            uti.append(((((i-1)*10+5)-eti[i-1])**2)*l[i-1])
            for x in range(t+1,maxvalue+1):
                c+=(((x*10+5)-ei[i-1])**2)*l[x]
            ui.append(c)
        elif 1<i<maxvalue+1:
            c=0
            for q in range(0,t):
                c+=(((q*10+5)-et[i-1])**2)*l[q]
            ut.append(c)
            c=0
            for x in range(t,i):
                c+=(((x*10+5)-eti[i-1])**2)*l[x]
            uti.append(c)
            c=0
            for y in range(i,maxvalue+1):
                c+=(((y*10+5)-ei[i-1])**2)*l[y]
            ui.append(c)
        else:
            c = 0
            for q in range(0, t):
                c += (((q * 10 + 5) - et[i - 1]) ** 2) * l[q]
            ut.append(c)
            c = 0
            for x in range(t,i):
                c+=(((x * 10 + 5) - et[i - 1]) ** 2) * l[x]
            uti.append(c)
            ui.append(0)

    for i in range(0,maxvalue+1):
        if i==0:
            sigama.append(((eti[i]-e)**2)*pti[i]+((ei[i]-e)**2)*pi[i])
        elif 0<i<maxvalue:
            sigama.append(((et[i]-e)**2)*pt[i]+((eti[i]-e)**2)*pti[i]+((ei[i]-e)**2)*pi[i])
        else:
            sigama.append(((et[i] - e) ** 2) * pt[i] + ((eti[i] - e) ** 2) * pti[i])

    high=(sigama.index(min(sigama))+1)*10
    low=high//2
    return high,low


def OneDimensionStandardNormalDistribution(x,sigma):
    E = -0.5/(sigma*sigma)
    return 1/(math.sqrt(2*math.pi)*sigma)*math.exp(x*x*E)


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img

def imcanny(pic_path):
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    pics = os.listdir(pic_path)

    for i in pics:
        if i[-4:] == '.png'or i[-4:] == '.jpg' or i[-5:] == '.jpeg':
            filename = pic_path + i
            img = plt.imread(filename)
            if i[-4:] == '.png':
                img = img*255
            img = img.mean(axis=-1)

            img2=cv2.bilateralFilter(img,5,50,50)

            plt.imshow(img2.astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.show()

            img3 = convolve(sobel_kernel_x,img2,[1,1,1,1],[1,1])
            img4 = convolve(sobel_kernel_y,img2,[1,1,1,1],[1,1])

            gradiant_length = (img3**2+img4**2)**(1.0/2)

            img3 = img3.astype(np.float64)
            img4 = img4.astype(np.float64)
            img3[img3==0]=0.00000001
            gradiant_tangent = img4/img3

            plt.imshow(gradiant_length.astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.show()


            final_img = DecideAndConnectEdge(gradiant_length,gradiant_tangent)

            cv2.imshow('edge',final_img.astype(np.uint8))

            cv2.waitKey(0)

def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
