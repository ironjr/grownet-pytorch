import math


class ColorMap(object):
    def __init__(self, fname):
        super(ColorMap, self).__init__()
        self.fname = fname
        with open(fname) as f:
            self.lines = f.readlines()

    def get_colors(self, vals):
        '''Get proper color in RGB form from its value

        Arguments:
            vals (float): list of or single float value between 0 and 1

        Returns:
            colors (str): list of or single RGB color with form "#%2x%2x%2x"
        '''
        nlines = len(self.lines)
        if not isinstance(vals, (list, tuple)):
            vals = [vals,]

        colors = []
        for val in vals:
            if val < 0 or val > 1:
                raise ValueError('value {} not in the interval [0, 1]'.format(val))
            val *= nlines - 1
            val_rem = val - math.floor(val)
            color_low = self.lines[math.floor(val)]
            color_high = self.lines[math.ceil(val)]
            color = self._interpolate(color_low, color_high, val_rem)
            colors.append(color)
        
        if len(colors) == 1:
            return colors[0]
        else:
            return colors

    def _str2rgb(self, color):
        '''String type RGB 256 color to int list type color

        Arguments:
            color (str): single RGB color with form "#%2x%2x%2x"

        Returns:
            r (int): R color value between 0 and 255
            g (int): G color value between 0 and 255
            b (int): B color value between 0 and 255
        '''
        r = int(color[1:3], base=16)
        g = int(color[3:5], base=16)
        b = int(color[5:], base=16)
        return r, g, b

    def _interpolate(self, color_a, color_b, val, mode='linear'):
        '''String type RGB 256 color interpolation

        Arguments:
            color_a (str): single RGB color with form "#%2x%2x%2x"
            color_b (str): single RGB color with form "#%2x%2x%2x"
            val (float): interpolation point between a(0) and b(1)
            mode (str): interpolation mode TODO

        Returns:
            color_i (str): single RGB color with form "#%2x%2x%2x"
        '''
        res = '#'
        for a, b in zip(self._str2rgb(color_a), self._str2rgb(color_b)):
            i = int(val * b + (1 - val) * a)
            i = max(0, min(255, i))
            res += '{:02x}'.format(i)
        return res
