import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Drawer(object):
    def __init__(self, x=32, y=32, z=5, agents_num=None, render=False):
        assert agents_num is not None
        self.n = agents_num
        self.x_range = x
        self.y_range = y
        self.z_range = z
        if not render:
            pass
        self.has_show = False

        fig = plt.figure(dpi=50, figsize=(16, 12))
        fig.add_subplot
        fig.suptitle('')

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title('Under training')
        ax.set_xlabel("Reward:")
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(0, self.y_range)
        ax.set_zlim(0, self.z_range)

        self.fig = fig
        self.ax = ax
        self.lines = [
            self.ax.plot3D([0, 1], [0, 1], [0, 1], c='slategrey', linewidth=5, marker='o', markerfacecolor='black')[0]
            for _ in range(self.n)]

    def set_data(self, data, rew):
        for line, dat in zip(self.lines, data):
            dat = [[li[0] for li in dat], [li[1] for li in dat], [li[2] for li in dat]]
            line.set_data(dat[0:2])
            line.set_3d_properties(dat[2])
        self.ax.set_xlabel("Reward: {.3f}".format(rew))
        return self.lines

    def show(self):
        if not self.has_show:
            plt.ion()
            self.has_show = True
        else:
            plt.pause(0.1)
