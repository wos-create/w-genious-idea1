"""
 Reference    :
 ReferenceAuthor:
 Year         :
              :
 Descripttion : 入射光束倾斜造成指向误差
 version      :
 Author       : ChenZhao
 Date         : 2024-07-13 17:13:56
 LastEditors  : ChenZhao
 LastEditTime : 2024-07-13 17:13:56
"""

import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib import font_manager # type: ignore
from matplotlib import rcParams # type: ignore

"""
全局字体显示设置 https://www.jianshu.com/p/ef35d42d47cc
中英文字体混合显示设置 https://zhuanlan.zhihu.com/p/501395717
"""

print(matplotlib.matplotlib_fname())  # 字体文件夹

# 字体加载
# font_path = "D:\\Anaconda3\\envs\\My_Matplotlib\\Lib\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\times+simsun.ttf"
font_path = "E:\\Download\\IncidentErrorScan\\times+simsun.ttf"

font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
print(prop.get_name())  # 显示当前使用字体的名称

# 字体设置
plt.rcParams["font.family"] = "sans-serif"  # 使用字体中的无衬线体
plt.rcParams["font.sans-serif"] = prop.get_name()  # 根据名称设置字体
plt.rcParams["axes.unicode_minus"] = False  # 使坐标轴刻度标签正常显示正负号
plt.rcParams["mathtext.fontset"] = "stix"  # 设置数学公式字体为stix

import numpy as np
from math import radians, degrees, sqrt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
from math import sin, cos, acos, atan, atan2, pi

# K9玻璃，楔角10°，折射系数1.5195
alpha1 = radians(10)
alpha2 = radians(10)
n0 = 1.0  # 空气折射率
n1 = 1.5195
n2 = 1.5195
Pixel_size = 5.5e-3  # 像元尺寸


# 入射光束误差（用旋转轴和旋转角定义倾斜）
def modelfun_withIncidentError(solutions, states):
    delta_x = solutions[0]
    delta_y = solutions[1]
    delta_z = solutions[2]
    betaO = solutions[3]
    delta = solutions[4]

    betaO_rad = radians(betaO)
    delta_rad = radians(delta)

    I = np.eye(3)

    k_t1 = np.matrix([cos(betaO_rad), sin(betaO_rad), 0])  # 旋转轴单位向量
    K_t1 = np.matrix([[0, -k_t1[0, 2], k_t1[0, 1]], [k_t1[0, 2], 0, -k_t1[0, 0]], [-k_t1[0, 1], k_t1[0, 0], 0]])
    temp = K_t1 * K_t1
    Rodrigues_Matrix_t1 = I + sin(delta_rad) * K_t1 + (1 - cos(delta_rad)) * (K_t1 * K_t1)  # Rodrigues 罗德里格斯矩阵
    eulerRad_z = atan2(Rodrigues_Matrix_t1[1, 0], Rodrigues_Matrix_t1[0, 0])
    eulerRad_y = atan2(-Rodrigues_Matrix_t1[2, 0], 1)
    eulerRad_x = atan2(Rodrigues_Matrix_t1[2, 1], Rodrigues_Matrix_t1[2, 2])

    theta1_deg = states[0]
    theta2_deg = states[1]
    dPixel_x = states[2]
    dPixel_y = states[3]
    focus_length = 32
    dis_target = 205

    theta1_rad = radians(theta1_deg)
    theta2_rad = radians(theta2_deg)

    if (not np.any(solutions)) & (not np.any([states[2], states[3]])):
        Sr0_x = 0
        Sr0_y = 0
        Sr0_z = 1
    else:
        # 欧拉角获取旋转矩阵
        # eulerRad_x = radians(eulerDeg_x)
        # eulerRad_y = radians(eulerDeg_y)
        # eulerRad_z = radians(eulerDeg_z)
        eul = [eulerRad_z, eulerRad_y, eulerRad_x]  # 分别是绕Z轴、Y轴、X轴的旋转角度（弧度）
        rotation = R.from_euler("ZYX", eul, degrees=False)  # 创建一个Rotation对象
        RotMatrix = rotation.as_matrix()

        light_Vec = np.array([[dPixel_x * Pixel_size], [dPixel_y * Pixel_size], [focus_length]])  # 光线向量
        Rotlight_Vec = np.transpose(np.dot(np.transpose(light_Vec), RotMatrix))  # 右乘绕自身轴
        Rotlight_UnitVec = normalize(Rotlight_Vec, "l2", axis=0)  # # L2范数正则化

        # 目标光线入射向量(单位向量)
        Sr0_x = Rotlight_UnitVec[0]
        Sr0_y = Rotlight_UnitVec[1]
        Sr0_z = Rotlight_UnitVec[2]


    # 棱镜1左平面经过点坐标
    Or0_x = 0
    Or0_y = 0
    Or0_z = 0
    # 世界坐标系定义在入射面上
    # 棱镜1左平面单位法向量（顺光线方向定义）
    Nr0_x = 0
    Nr0_y = 0
    Nr0_z = 1

    # 棱镜1左楔面与传播光线的交点（xr0，yr0，zr0）
    t0 = (Nr0_x * (Or0_x - delta_x) + Nr0_y * (Or0_y - delta_y) + Nr0_z * (Or0_z - (-37.8142 + delta_z))) / (Nr0_x * Sr0_x + Nr0_y * Sr0_y + Nr0_z * Sr0_z)
    xr0 = t0 * Sr0_x + delta_x
    yr0 = t0 * Sr0_y + delta_y
    zr0 = t0 * Sr0_z + (-37.8142 + delta_z)

    # 棱镜1右楔面经过点坐标
    Or1_x = 0
    Or1_y = 0
    Or1_z = 20.3429
    # 棱镜1右楔面单位法向量（顺光线方向定义）
    Nr1_x = -cos(theta1_rad) * sin(alpha1)
    Nr1_y = -sin(theta1_rad) * sin(alpha1)
    Nr1_z = cos(alpha1)

    # 棱镜2左楔面经过点坐标
    Or2_x = 0
    Or2_y = 0
    Or2_z = 81.0287
    # 棱镜2左楔面单位法向量（顺光线方向定义）
    Nr2_x = cos(theta2_rad) * sin(alpha2)
    Nr2_y = sin(theta2_rad) * sin(alpha2)
    Nr2_z = cos(alpha2)

    # 棱镜2右平面经过点坐标
    Or3_x = 0
    Or3_y = 0
    Or3_z = 101.3716
    # 棱镜2右平面单位法向量（顺光线方向定义）
    Nr3_x = 0
    Nr3_y = 0
    Nr3_z = 1

    # 光屏平面经过点坐标
    Orp_x = 0
    Orp_y = 0
    Orp_z = 101.3716 + dis_target
    # 代表光屏距离棱镜2右平面dis_target mm
    # 光屏平面单位法向量（顺光线方向定义）
    Nrp_x = 0
    Nrp_y = 0
    Nrp_z = 1

    # 棱镜1左平面入射光线单位向量 Sr0
    # 经棱镜1左平面折射后的折射光线单位向量 Sr1
    Sr1 = [
        (n0 * (Sr0_x - Nr0_x * (Sr0_x * Nr0_x + Sr0_y * Nr0_y + Sr0_z * Nr0_z))) / n1 + Nr0_x * sqrt((n0**2 * (Sr0_x * Nr0_x + Sr0_y * Nr0_y + Sr0_z * Nr0_z) ** 2) / n1**2 - n0**2 / n1**2 + 1),
        (n0 * (Sr0_y - Nr0_y * (Sr0_x * Nr0_x + Sr0_y * Nr0_y + Sr0_z * Nr0_z))) / n1 + Nr0_y * sqrt((n0**2 * (Sr0_x * Nr0_x + Sr0_y * Nr0_y + Sr0_z * Nr0_z) ** 2) / n1**2 - n0**2 / n1**2 + 1),
        (n0 * (Sr0_z - Nr0_z * (Sr0_x * Nr0_x + Sr0_y * Nr0_y + Sr0_z * Nr0_z))) / n1 + Nr0_z * sqrt((n0**2 * (Sr0_x * Nr0_x + Sr0_y * Nr0_y + Sr0_z * Nr0_z) ** 2) / n1**2 - n0**2 / n1**2 + 1),
    ]
    Sr1_x = Sr1[0]
    Sr1_y = Sr1[1]
    Sr1_z = Sr1[2]

    # 棱镜1左平面入射点坐标 (xr0,yr0,zr0)
    # 棱镜1右楔面经过点坐标 (Or1_x,Or1_y,Or1_z)
    # 棱镜1右楔面单位法向量（Nr1_x,Nr1_y,Nr1_z）
    # 经棱镜1左平面折射后的折射光线单位向量（Sr1_x,Sr1_y,Sr1_z）
    t1 = (Nr1_x * (Or1_x - xr0) + Nr1_y * (Or1_y - yr0) + Nr1_z * (Or1_z - zr0)) / (Nr1_x * Sr1_x + Nr1_y * Sr1_y + Nr1_z * Sr1_z)

    # 棱镜1右楔面与传播光线的交点（xr1，yr1，zr1）
    xr1 = t1 * Sr1_x + xr0
    yr1 = t1 * Sr1_y + yr0
    zr1 = t1 * Sr1_z + zr0

    # 经棱镜1右楔面折射后的折射光线单位向量 Sr2
    Sr2 = [
        (n1 * (Sr1_x - Nr1_x * (Sr1_x * Nr1_x + Sr1_y * Nr1_y + Sr1_z * Nr1_z))) / n0 + Nr1_x * sqrt((n1**2 * (Sr1_x * Nr1_x + Sr1_y * Nr1_y + Sr1_z * Nr1_z) ** 2) / n0**2 - n1**2 / n0**2 + 1),
        (n1 * (Sr1_y - Nr1_y * (Sr1_x * Nr1_x + Sr1_y * Nr1_y + Sr1_z * Nr1_z))) / n0 + Nr1_y * sqrt((n1**2 * (Sr1_x * Nr1_x + Sr1_y * Nr1_y + Sr1_z * Nr1_z) ** 2) / n0**2 - n1**2 / n0**2 + 1),
        (n1 * (Sr1_z - Nr1_z * (Sr1_x * Nr1_x + Sr1_y * Nr1_y + Sr1_z * Nr1_z))) / n0 + Nr1_z * sqrt((n1**2 * (Sr1_x * Nr1_x + Sr1_y * Nr1_y + Sr1_z * Nr1_z) ** 2) / n0**2 - n1**2 / n0**2 + 1),
    ]
    Sr2_x = Sr2[0]
    Sr2_y = Sr2[1]
    Sr2_z = Sr2[2]

    # 棱镜1右楔面入射点坐标 (xr1,yr1,zr1)
    # 棱镜2左楔面经过点坐标 (Or2_x,Or2_y,Or2_z)
    # 棱镜2左楔面单位法向量（Nr2_x,Nr2_y,Nr2_z）
    # 经棱镜1右楔面折射后的折射光线单位向量（Sr2_x,Sr2_y,Sr2_z）
    t2 = (Nr2_x * (Or2_x - xr1) + Nr2_y * (Or2_y - yr1) + Nr2_z * (Or2_z - zr1)) / (Nr2_x * Sr2_x + Nr2_y * Sr2_y + Nr2_z * Sr2_z)

    # 棱镜2左楔面与传播光线的交点（xr2，yr2，zr2）
    xr2 = t2 * Sr2_x + xr1
    yr2 = t2 * Sr2_y + yr1
    zr2 = t2 * Sr2_z + zr1

    # 经棱镜2左楔面折射后的折射光线单位向量 Sr3
    Sr3 = [
        (n0 * (Sr2_x - Nr2_x * (Sr2_x * Nr2_x + Sr2_y * Nr2_y + Sr2_z * Nr2_z))) / n2 + Nr2_x * sqrt((n0**2 * (Sr2_x * Nr2_x + Sr2_y * Nr2_y + Sr2_z * Nr2_z) ** 2) / n2**2 - n0**2 / n2**2 + 1),
        (n0 * (Sr2_y - Nr2_y * (Sr2_x * Nr2_x + Sr2_y * Nr2_y + Sr2_z * Nr2_z))) / n2 + Nr2_y * sqrt((n0**2 * (Sr2_x * Nr2_x + Sr2_y * Nr2_y + Sr2_z * Nr2_z) ** 2) / n2**2 - n0**2 / n2**2 + 1),
        (n0 * (Sr2_z - Nr2_z * (Sr2_x * Nr2_x + Sr2_y * Nr2_y + Sr2_z * Nr2_z))) / n2 + Nr2_z * sqrt((n0**2 * (Sr2_x * Nr2_x + Sr2_y * Nr2_y + Sr2_z * Nr2_z) ** 2) / n2**2 - n0**2 / n2**2 + 1),
    ]
    Sr3_x = Sr3[0]
    Sr3_y = Sr3[1]
    Sr3_z = Sr3[2]

    # 棱镜2左楔面入射点坐标 (xr2,yr2,zr2)
    # 棱镜2右平面经过点坐标 (Or3_x,Or3_y,Or3_z)
    # 棱镜2右平面单位法向量（Nr3_x,Nr3_y,Nr3_z）
    # 经棱镜2左楔面折射后的折射光线单位向量（Sr3_x,Sr3_y,Sr3_z）
    t3 = (Nr3_x * (Or3_x - xr2) + Nr3_y * (Or3_y - yr2) + Nr3_z * (Or3_z - zr2)) / (Nr3_x * Sr3_x + Nr3_y * Sr3_y + Nr3_z * Sr3_z)

    # 棱镜2右平面与传播光线的交点（xr3，yr3，zr3）
    xr3 = t3 * Sr3_x + xr2
    yr3 = t3 * Sr3_y + yr2
    zr3 = t3 * Sr3_z + zr2

    # 经棱镜2右平面折射后的折射光线单位向量 Sr4
    Srp = [
        (n2 * (Sr3_x - Nr3_x * (Sr3_x * Nr3_x + Sr3_y * Nr3_y + Sr3_z * Nr3_z))) / n0 + Nr3_x * sqrt((n2**2 * (Sr3_x * Nr3_x + Sr3_y * Nr3_y + Sr3_z * Nr3_z) ** 2) / n0**2 - n2**2 / n0**2 + 1),
        (n2 * (Sr3_y - Nr3_y * (Sr3_x * Nr3_x + Sr3_y * Nr3_y + Sr3_z * Nr3_z))) / n0 + Nr3_y * sqrt((n2**2 * (Sr3_x * Nr3_x + Sr3_y * Nr3_y + Sr3_z * Nr3_z) ** 2) / n0**2 - n2**2 / n0**2 + 1),
        (n2 * (Sr3_z - Nr3_z * (Sr3_x * Nr3_x + Sr3_y * Nr3_y + Sr3_z * Nr3_z))) / n0 + Nr3_z * sqrt((n2**2 * (Sr3_x * Nr3_x + Sr3_y * Nr3_y + Sr3_z * Nr3_z) ** 2) / n0**2 - n2**2 / n0**2 + 1),
    ]
    Srp_x = Srp[0]
    Srp_y = Srp[1]
    Srp_z = Srp[2]

    # 棱镜2右平面入射点坐标 (xr3,yr3,zr3)
    # 光屏平面经过点坐标 (Orp_x,Orp_y,Orp_z)
    # 光屏平面单位法向量（Nrp_x,Nrp_y,Nrp_z）
    # 经棱镜2右平面折射后的折射光线单位向量（Sr4_x,Sr4_y,Sr4_z）
    t4 = (Nrp_x * (Orp_x - xr3) + Nrp_y * (Orp_y - yr3) + Nrp_z * (Orp_z - zr3)) / (Nrp_x * Srp_x + Nrp_y * Srp_y + Nrp_z * Srp_z)

    # 光屏与传播光线的交点（xr4，yr4，zr4）
    xrp = t4 * Srp_x + xr3
    yrp = t4 * Srp_y + yr3
    zrp = t4 * Srp_z + zr3

    return xr3, yr3, zr3, xrp, yrp, zrp, Srp_x, Srp_y, Srp_z


resolution_deg = 1  # 棱镜转角分辨率10°
number = 360 / resolution_deg

theta1_num = number  # 棱镜转角1采样点数
theta2_num = number  # 棱镜转角2采样点数

theta1_deg = np.arange(0, 360, resolution_deg)
theta2_deg = np.arange(0, 360, resolution_deg)

D = 205  # 与目标面的垂直距离（实际测量）
dis_camera = 50
delta_x = 0
delta_y = 0
delta_z = -dis_camera
eulerDeg_x = 0.3  #
eulerDeg_y = 0.3  #
eulerDeg_z = 0  #

error_parameter = [-0.2139753, 0.92990323, -20.81066707, 90, 1]

fig1, ax1 = plt.subplots()
X_delta0 = []
Y_delta0 = []

for j in range(0, int(number), 1):
    result = modelfun_withIncidentError(error_parameter, [theta1_deg[j], theta2_deg[j], 0, 0])
    X_delta0.append(result[3])
    Y_delta0.append(result[4])

plt.plot(X_delta0, Y_delta0, c="b")
plt.axvline(x=0, color="r", linestyle="--")
plt.axhline(y=0, color="r", linestyle="--")

plt.scatter(0, 0, color="r")
plt.xlim(-80, 80)
plt.ylim(80, -80)
# 设置x和y轴的比例为1
ax1.set_aspect("equal")

plt.show()
