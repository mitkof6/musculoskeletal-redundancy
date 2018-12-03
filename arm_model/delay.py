#!/usr/bin/env python

import numpy as np
import pylab as pl
import unittest
from logger import Logger
from scipy.interpolate import CubicSpline


class Delay:
    """Implements a signal delay.

    We assume that values prior to the delay have a default value (y(t < t_c -
    d) = v). Moreover we define a memory variable that is 10 x delay and
    restricts the size of the buffer.

    """

    def __init__(self, delay, default_value):
        """
        1D Delay

        Parameters
        ----------
        delay: the delay of this component
        default_value: the default value of the delay

        """
        self.logger = Logger('Delay')
        self.t = []
        self.y = []
        self.delay = delay
        self.memory = 1000.0 * delay
        self.default_value = default_value
        self.assert_add = False

    def add(self, t, y):
        """Append the delay buffer with the current value of the signal.

        Restrict the buffer to contain values coresponding to:

        [current time - memory (K x delay)], K~1000

        Parameters
        ----------

        t: time
        y: value

        """
        # ensure that time is in the range [t - memory, t]
        time = np.array(self.t)
        values = np.array(self.y)
        mask = (np.array(time) < t) & (np.array(time) > t - self.memory)
        self.t = time[mask].tolist()
        self.y = values[mask].tolist()

        # append container
        self.t.append(t)
        self.y.append(y)

        self.assert_add = True

    def get_delayed(self):
        """Get a delaied version of the signal (CubicSpline). Ensure to call add(t, y)
        before getting a delayed value.

        Returns
        -------

        a delayed version of the signal y

        """
        assert self.assert_add == True, 'Should call add(t, y) before get_delayed()'

        t = self.t
        y = self.y
        d = self.delay

        # # 2 (this can cause problem during numerical integration)
        # if len(t) == 2 and t[-1] - d >= 0:
        #     return y[0] + (y[1] - y[0]) / (t[1] - t[0]) * (d - t[0])

        # < 3
        if len(t) < 3 or t[-1] - d < 0:
            return self.default_value

        # 3+
        cs = CubicSpline(np.array(t), np.array(y))

        self.assert_add = False

        return cs(t[-1] - d)


class DelayArray:
    """
    Implements a N-D signal delay.

    We assume that values prior to the delay have a default value (y(t < t_c -
    d) = v). Moreover we define a memory variable that is 10 x delay and
    restricts the size of the buffer.

    """

    def __init__(self, n, delay, default_value):
        """
        N-D Delay

        Parameters
        ----------
        delay: n x 1 array of delays
        default_value: n x 1 array of default values

        """
        self.n = n
        self.delay_array = [Delay(delay[i], default_value[i])
                            for i in range(n)]

    def add(self, t, y):
        """Append the delay buffer with the current value of the signal.

        Restrict the buffer to contain values coresponding to:

        [current time - memory (10.0 x delay)]

        Parameters
        ----------
        t: time
        y: n x 1 array of values

        """

        n = self.n
        assert len(y) == n, 'Dimensions mismatch in y'

        [self.delay_array[i].add(t, y[i]) for i in range(n)]

    def get_delayed(self):
        """Get a delaied version of the signal (CubicSpline). Ensure to call add(t, y)
        before getting a delayed value.

        Returns
        -------
        a delayed version of the signal y

        """
        return [self.delay_array[i].get_delayed() for i in range(self.n)]


class TestDelay(unittest.TestCase):

    def test_delay(self):
        d = np.pi / 2
        delay = Delay(d, 0.2)
        t = np.linspace(0, 2.5 * np.pi, num=100, endpoint=True)
        y = []
        yd = []
        for i in t:
            y.append(np.sin(i) + 0.1 * np.cos(7 * i))
            delay.add(i, y[-1])
            yd.append(delay.get_delayed())

        # plot
        pl.figure()
        pl.plot(t, y, 'r', t, yd, 'b')
        pl.title('Delay = ' + str(d))
        pl.xlabel('$t \; (s)$')
        pl.ylabel('$y(t)$')
        pl.legend(['$y(t)$', '$y(t-d)$'])

    def test_delay_array(self):
        n = 2
        delay = [np.pi / 2, np.pi / 4]
        default_value = [0.1, 0.2]
        delay_array = DelayArray(2, delay, default_value)
        t = np.linspace(0, 2.5 * np.pi, num=100, endpoint=True)
        y = []
        yd = []
        for i in t:
            y1 = np.sin(i) + 0.1 * np.cos(7 * i)
            y2 = np.sin(i) - 0.1 * np.cos(7 * i)
            y.append([y1, y2])
            delay_array.add(i, y[-1])
            yd.append(delay_array.get_delayed())

        # plot
        pl.figure()
        pl.plot(t, np.array(y), 'r', t, np.array(yd), 'b')
        pl.title('Delay = ' + str(delay))
        pl.xlabel('$t \; (s)$')
        pl.ylabel('$y(t)$')
        pl.legend(['$y(t)$', '$y(t-d)$'])


if __name__ == '__main__':
    unittest.main()
