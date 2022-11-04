import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle


class MyRArrow(BoxStyle.RArrow):
    def __init__(self, pad=0.3, width=220):
        self.width_ = width
        super(MyRArrow, self).__init__(pad)

    def transmute(self, x0, y0, width, height, mutation_size):
        p = BoxStyle.RArrow.transmute(self, x0, y0,
                                      width, height, mutation_size)
        x = p.vertices[:, 0]
        p.vertices[1:3, 0] = x[1:3] - self.width_
        p.vertices[0, 0] = x[0] + self.width_
        p.vertices[3:, 0] = x[3:] + self.width_
        return p
BoxStyle._style_list["myrarrow"] = MyRArrow


if __name__ == '__main__':
    measures = [0.2, -0.5]
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes([0, 0, 1, 1])
    plt.plot(list(range(10)), list(range(10)))
    # x axis arrow
    plt.quiver(4, -1.0, 4, 0, scale=8, width=0.02)  # scale 8 to 20
    plt.show()

