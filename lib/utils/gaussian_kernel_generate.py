import numpy as np
import matplotlib.pyplot as plt

# Give an odd number
KERNEL_SIZE = 255

def visualGaussianWeight(weightMap, filename):
    fig = plt.imshow(weightMap, cmap="jet")
    plt.axis('on') # off
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print('Save image: {0}'.format(filename))

def Gaussain2D(x, y, th, A, x0, y0, sigma_x, sigma_y):
    a = (np.cos(th)**2) / (2 * sigma_x**2) + \
        (np.sin(th)**2) / (2 * sigma_y**2)
    b = -(np.sin(2*th)) / (4 * sigma_x**2) + \
         (np.sin(2*th)) / (4 * sigma_y**2)
    c = (np.sin(th)**2) / (2 * sigma_x**2) + \
        (np.cos(th)**2) / (2 * sigma_y**2)
    z = A * np.exp( -(a * (x - x0)**2 - \
                      2 * b * (x - x0) * (y - y0) + \
                      c * (y - y0)**2
                     ) 
                  )
    return z

def sumDirectionMaxValue(Z, r, s):
    sum = 0.0
    if r == 0:
        for i in xrange(s*2+1):
            if i != s:
                sum += Z[s][i]
    elif r == 45:
        i = s*2+1
        for j in xrange(s*2+1):
            i -= 1
            if i != j:
                sum += Z[i][j]
    elif r == 90:
        for j in xrange(s*2+1):
            if j != s:
                sum += Z[j][s]
    elif r == 135:
        i = 0
        for j in xrange(s*2+1):
            if i != s and j != s:
                sum += Z[i][j]
            i += 1
    #print Z
    return sum

def Gaussain_kernel_generate(kernel_size):
    A = 1; # Amplitude
    x0 = 0; y0 = 0; # whole function shifts
    theta = np.arange(0, np.pi, np.pi/4) # rotation
    
    half_szie = int((kernel_size - 1) / 2)
    start = -half_szie; end = -start; step = 1;
    
    sigma_x = half_szie / 8; sigma_y = half_szie / 1.1; # scatter degree

    # Generate (X, Y) coordinates
    shift_x = np.arange(start, end+1, step)
    shift_y = np.arange(start, end+1, step)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) 
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
   
    # Generate Gaussain distribution at each position (X, Y)
    all_Z = []
    all_norm_value = []
    for th in theta:
        rotate =  int(th * 180 / np.pi)
        Z = np.zeros( (len(shift_x), len(shift_y)) )
        for [x, y] in shifts:
            z = Gaussain2D(x, y, th, A, x0, y0, sigma_x, sigma_y)
            Z[x + end][y + end] = z
        all_Z.append(Z)
        #visualGaussianWeight(Z, "./gaussian_weight_{0}.jpg".format(str(th)))
        #norm_value = sumDirectionMaxValue(Z, rotate, -start)
        norm_value = np.sum(Z)
        all_norm_value.append(norm_value)

    return np.array(all_Z), np.array(all_norm_value)

def main():
    all_Z, all_norm_value = Gaussain_kernel_generate(KERNEL_SIZE)
    for Z in all_Z:
        print('====================')
        print(Z)


if __name__ == '__main__':
    main()
