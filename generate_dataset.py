#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:39 2020

@author: blyu
"""

# python txt2hidden_states_yi.py /sshare/home/lyubingjiang/lm/corpus/ptb/ptb3-wsj-23  iter_0010000  0      1
#                                input_txt_file                                       model         layer  gpu_id

import torch, os, string, time
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

wd_id_droppeds_1 = [
    [4094, 4095, 4096, 4097, 4098, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4352, 4353, 4354, 4355, 4356, 5022, 5023, 5024, 5025, 5026, 5027, 5031, 5032, 5033, 5034, 5035, 5036, 5037],
    [1953, 1954, 1955, 1956, 1957, 1958, 1959, 2435, 2436, 2437, 2438, 2439, 2440, 3424, 3425, 3426, 3427, 7776, 7777, 7778, 7779, 7780, 7781, 7782, 7783, 7784, 7785, 7786, 7787],
    [53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 205, 206, 207, 208, 209, 265, 266, 267, 268, 770, 771, 772, 773, 774, 775, 2821, 2822, 2823, 2824, 2825, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5257, 5258, 5259, 5260, 5261, 5262, 5263, 5264, 5265, 5285, 5286, 5287, 5288, 5289, 5366, 5367, 5368, 5369, 5370, 5371, 5372, 5373, 5374, 5375, 5376, 5377, 5378, 5379, 5380, 5381, 5382, 5383, 5384, 5385, 5386, 5387, 5388, 5389, 5390, 5391, 5392, 5393, 5762, 5763, 5764, 5765, 5766, 5767, 5768, 5769, 5770, 6255, 6256, 6257, 6258, 6259, 6260, 6261],
    [992, 993, 994, 995, 996, 997, 998, 999, 1000, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2676, 2677, 2678, 2679, 2680, 2681, 3912, 3913, 3914, 3915, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987],
    [736, 737, 738, 739, 740, 741, 1218, 1219, 1220, 1415, 1416, 1417, 1418, 1419, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2504, 2505, 2506, 2507, 2508, 2509, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2590, 2591, 2592, 3260, 3261, 3262, 3263, 3264, 3265, 3553, 3554, 3555, 3556, 4722, 4723, 4724, 4725, 4726, 4727, 4728, 4729, 5389, 5390, 5391, 5930, 5931, 5932, 5933, 5984, 5985, 5986, 5987, 5988, 5989, 6694, 6695, 6696, 6697, 6698, 6699, 6700, 6701, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266],
    [2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2831, 2832, 2833, 2834, 2835, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 3184, 3185, 3186, 3187, 3188, 3189, 4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 5493, 5494, 5495, 5496, 5497, 5498, 7571, 7572, 7573, 7574, 7575, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8015, 8016],
    [294, 295, 296, 297, 298, 760, 761, 762, 763, 764, 1239, 1240, 1241, 1242, 1243, 1244, 2341, 2342, 2343, 2344, 3347, 3348, 3349, 3350, 3351, 3584, 3585, 3586, 3587, 3588, 4446, 4447, 4448, 4449, 4450],
    [2552, 2553, 2554, 2555, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2856, 2857, 2858, 2859, 2860, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3882, 3883, 3884, 3885, 3886, 6485, 6486, 6487, 6488, 6489, 6490, 6495, 6496, 6497, 6498, 6499, 6500, 6501, 6502, 6503, 7801, 7802, 7803, 7804, 7805],
    [184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 1744, 1745, 1746, 1747, 1748, 1749, 1865, 1866, 1867, 1868, 1869, 1870, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2747, 2748, 2749, 2750, 6751, 6752, 6753, 6754, 6755, 6756, 6757, 6758, 6759, 6760, 6761, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6769, 6770, 6771, 6772, 6773, 6774, 6775, 6776, 6777, 6778],
    [5107, 5108, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5116, 6812, 6813, 6814, 6815, 6816, 6817, 6818, 6824, 6825, 6826, 6827, 6828, 6829, 6830, 6831, 7996, 7997, 7998, 7999, 8000, 8001, 8002, 8003, 8004, 8005]
]

wd_id_droppeds_2 = [
    [283, 284, 285, 286, 287, 288, 291, 292, 293, 294, 295, 296, 297, 298, 517, 518, 519, 520, 521, 522, 523, 524, 525, 4669, 4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4894, 4895, 4896, 4897, 5700, 5701, 5702, 5703, 5704, 5705, 6460, 6461, 6462, 6463, 6464],
    [970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 1383, 1384, 1385, 1386, 1387, 1388, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 2763, 2764, 2765, 2766, 2767, 2879, 2880, 2881, 2882, 2883, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 9153, 9154, 9155, 9156, 9157, 9158],
    [612, 613, 614, 615, 616, 617, 618, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 5139, 5140, 5141, 5142, 5143, 5144, 5145, 5146, 5147, 5148, 5149, 5150, 5151, 5152, 5153, 5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5163, 5164],
    [1285, 1286, 1287, 1288, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 7349, 7350, 7351, 7352, 7353, 7354, 7355, 7356],
    [3176, 3177, 3178, 3179, 3180, 3181, 3572, 3573, 3574, 3575, 3576, 3577],
    [2657, 2658, 2659, 2660, 2661, 2662, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061, 5062, 5063, 5064, 5065, 5066, 5067, 5068, 5069],
    [1150, 1151, 1152, 1153, 1154, 1155, 1156, 4368, 4369, 4370, 4371, 4372, 4373, 4414, 4415, 4416, 4417, 4418, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 6630, 6631, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6639],
    [1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 3361, 3362, 3363, 3364, 3365, 3366, 4913, 4914, 4915, 4916, 4917, 4918, 4919, 4920, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930, 4931, 4932, 5073, 5074, 5075, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5083, 5084, 5085, 6028, 6029, 6030, 6031, 6032, 6033, 6034, 6035, 6036, 6037, 6038, 6039, 6040, 6041, 6042, 6043, 6044, 6045, 6046, 6047, 6048, 6049, 6050, 6051, 6052, 6053, 6054, 6055, 6056, 6529, 6530, 6760, 6761, 6762, 6763, 6764, 6765, 6766, 6767, 6768, 6769, 6770, 7936, 7937, 7938, 7939, 7940, 7941, 9241, 9242, 9243, 9244, 9245, 9246, 9247, 9248, 9249, 9250, 9251, 9252, 9253, 9254, 9255, 9256, 9257, 9258, 9259, 9260, 9261, 9262, 9263, 9264, 9265, 9266, 9267, 9268],
    [978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 2300, 2301, 2302, 2303, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 3356, 3357, 3358, 3359, 3360, 3361, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4796, 4797, 4798, 4799, 4800, 6903, 6904, 6905, 6909, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 6919, 6920, 6921, 6922, 6923, 6924, 6925, 6926, 6927, 6928, 6929, 6930, 6931, 6932, 6933, 6934, 6935, 6936, 6937, 6938, 6939, 6940, 6941, 6942, 6943, 6944, 6945, 6946, 8094, 8095, 8096, 8097, 8098, 8099, 8100, 8101],
    [7, 8, 9, 10, 11, 12, 13, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 130, 131, 132, 133, 134, 135, 136, 183, 184, 185, 186, 187, 188, 191, 192, 229, 230, 231, 232, 233, 234, 251, 252, 253, 427, 428, 429, 430, 431, 432, 433, 434, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 794, 795, 796, 797, 798, 799, 800, 802, 803, 804, 805, 806, 807, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174]
]

wd_id_droppeds_3 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 304, 305, 306, 307, 308, 309, 310, 311, 312, 882, 883, 884, 885, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1442, 1443, 1444, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 2512, 2513, 2514, 2515, 2516, 2517, 2520, 2521, 2522, 2523, 2524, 2525, 2609, 2610, 2611, 2612, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2882, 2883, 2884, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3977, 3978, 3979, 3980, 3981, 3982, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4288, 4289, 4290, 4291, 4292, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4595, 4596, 4597, 4598, 4599, 4600, 4684, 4685, 4686, 4687, 4827, 4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4885, 4886, 4887, 4931, 4932, 4933, 4934, 4935, 4936, 5101, 5102, 5103, 5104, 5105, 5106, 6534, 6535, 6536, 6537, 6538, 6539, 6540, 6560, 6561, 6562, 6563, 6564, 6565, 6566, 6567, 6967, 6968, 6969, 6970, 6971, 6972, 6973, 7444, 7445, 7446, 7447, 7448],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 140, 141, 142, 143, 144, 145, 146, 147, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 179, 180, 181, 182, 183, 184, 185, 186, 212, 213, 214, 215, 216, 217, 218, 219, 220, 803, 804, 805, 806, 807, 808, 809, 810, 811, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 2123, 2124, 2125, 2126, 2127, 2128, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2583, 2584, 2585, 2586, 2587, 2588, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 5236, 5237, 5238, 5239, 5240, 5241, 5242, 5243, 5244, 5245, 5246, 5247, 5248, 5249, 5250, 5374, 5375, 5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 6100, 6101, 6102, 6103, 6104, 6105, 6106, 6107, 6114, 6115, 6116, 6117, 6118, 6119, 6120, 6121, 6122, 6959, 6960, 6961, 6962, 6963, 6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971, 8348, 8349, 8350, 8351, 8352, 8353, 8354, 9079, 9080, 9081, 9082, 9083, 9084, 9120, 9121, 9122, 9123, 9124, 9125, 9126, 9127, 9128, 9216, 9217, 9218, 9219],
    [289, 290, 291, 292, 293, 294, 295, 296, 297, 612, 613, 614, 615, 616, 617, 618, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 2275, 2276, 2277, 2278, 2279, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3657, 3658, 3659, 3660, 3661, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3669, 3670, 3671, 3672, 3673, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 5137, 5138, 5470, 5471, 5472, 5473, 5474, 5475, 5956, 5957, 5958, 5959, 5960, 5961, 5962, 5963, 5964, 5965, 5966, 5967, 6009, 6010, 6011, 6012, 6013, 6014, 6121, 6122, 6123, 6124, 6125, 6126, 6213, 6214, 6215, 6216, 6217, 6309, 6310, 6311, 6312, 6313, 6314, 6315, 6316, 6317, 6318, 6319, 6320, 6423, 6424, 6425, 6426, 6427, 6428, 6429, 6430, 6431, 6432, 6434, 6435, 6436, 6437, 6438, 6439, 6440, 6441, 6442, 6443, 6444, 6445, 6519, 6520, 6521, 6522, 6523, 6524, 6525, 6526, 6527, 6822, 6823, 6824, 6825, 6826, 6827, 6828, 6829, 6830, 6831, 6832, 6984, 6985, 6986],
    [0, 1, 2, 3, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1899, 1900, 6954, 6955, 6956, 6957, 6958, 6959, 6960, 6961, 6962, 6963, 6964, 7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299],
    [339, 340, 341, 342, 343, 344, 345, 346, 347, 378, 379, 380, 381, 382, 480, 481, 482, 483, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 3090, 3091, 3092, 3093, 3094, 3356, 3357, 3358, 3359, 3360, 3361, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3729, 3730, 3731, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 5878, 5879, 5880, 5881, 5882, 5883, 5884, 5885, 5886, 5887, 5888, 5889, 5890, 5891, 5892, 5893, 5894, 5895, 5896, 5897, 5898, 5899, 5900, 5901, 5902, 6311, 6312, 6313, 6314, 6315, 6316, 6317, 6318, 6319, 6320, 6323, 6324, 6325, 6326, 6327, 6328, 6329, 6330, 6331, 6332, 6333, 6972, 6973, 6974, 6975, 6976, 6977, 6978, 6979, 6980, 6981, 6982, 6983, 6984, 6995, 6996, 6997, 6998, 6999, 7000, 7001, 7002, 7003],
    [204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 2188, 2189, 2190, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 3168, 3169, 3170, 3171, 3172, 3173, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3821, 3822, 3823, 3824, 3825, 3826, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 6132, 6133, 6134, 6135, 6136, 6139, 6140, 6141, 6142, 6143, 6144, 6145, 6146, 6147, 7183, 7184, 7185, 7186, 7187, 9175, 9176, 9177],
    [759, 760, 761, 927, 928, 929, 930, 931, 932, 933, 934, 935, 1101, 1102, 1103, 1104, 1105, 1363, 1364, 1365, 1366, 1367, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2241, 2242, 2243, 2244, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2748, 2749, 2750, 2751, 2752, 2753, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2789, 2790, 2791, 2792, 2793, 2794, 3040, 3041, 3042, 3043, 3044, 3045, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3606, 3607, 3608, 3609, 3610, 3611, 3791, 3792, 3793, 3794, 3795, 3976, 3977, 3978, 3979, 4386, 4904, 4905, 4906, 5232, 5233, 5234, 5235, 5236, 5237, 5238, 5239, 5244, 5245, 5246, 5247, 5248, 5249, 5250, 5251, 5252, 5253, 5254, 5255, 5256, 5267, 5268, 5269, 5270, 5271, 5272, 5273, 5274, 5275, 5276, 5482, 5483, 5484, 5485, 5486, 6216, 6217, 6218, 6219, 7286, 7287, 7288, 7289, 7445, 7446, 7447, 7448, 7449, 7450, 7451, 7452, 7453, 7454],
    [165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 341, 342, 343, 344, 1614, 1615, 1616, 1617, 1618, 1619, 1895, 1896, 1897, 1898, 1899, 1900, 2547, 2548, 2549, 2550, 2551, 2552, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567],
    [322, 323, 324, 325, 326, 327, 328, 588, 589, 590, 591, 592, 593, 872, 873, 874, 875, 876, 877, 878, 879, 986, 987, 988, 989, 990, 991, 992, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2219, 2220, 2221, 2222, 2282, 2283, 2284, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2824, 2825, 2826, 2827, 2828, 2895, 2896, 2897, 2898, 2899, 5037, 5038, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5094, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 5583, 6605, 6606, 6607, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895, 7975, 7976, 7977, 7978, 7979, 7980, 7981, 7982, 7983, 7984, 7985],
    [442, 443, 444, 2599, 2600, 2601, 2602, 2603, 2604, 6728, 6729, 6730, 6731, 6732, 6733, 6734, 6735, 6736, 6739, 6740, 6741, 6742, 6743, 6744, 6745, 6746, 6747, 6748, 6749, 7711, 7712, 7713, 7714, 7715, 7716, 7717, 7718, 8113, 8114, 8115, 8116, 8117]
]

wd_id_ds = [[], wd_id_droppeds_1, wd_id_droppeds_2, wd_id_droppeds_3]

def maket(t):
    t = time.localtime(t)
    t = time.strftime("%Y-%m-%d %H:%M:%S", t)
    return t

def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):

    mapping = defaultdict(list)
    untokenized_sent_index = 0

    for tokenized_sent_index in range(len(tokenized_sent)):
        if tokenized_sent_index == 0:
          mapping[untokenized_sent_index].append(tokenized_sent_index)
          i1 = tokenized_sent_index
        else:
          tmp = ''.join(tokenized_sent[i1:tokenized_sent_index+1])
          #print(tmp, '\x20\x20▁'+untokenized_sent[untokenized_sent_index])
          if tmp in '▁'+untokenized_sent[untokenized_sent_index]:
              #print(1)
              mapping[untokenized_sent_index].append(tokenized_sent_index)
          else:
              #print(0)
              untokenized_sent_index += 1
              mapping[untokenized_sent_index].append(tokenized_sent_index)
              i1 = tokenized_sent_index
          #print(mapping)
              
    return mapping

argp = ArgumentParser()
argp.add_argument('--model_name', default='iter_0010000')
argp.add_argument('--layer', default='8')
argp.add_argument('--gpuid', default='0')
argp.add_argument('--display', default=False)
argp.add_argument('--show', default=True)
args = argp.parse_args()

model_name = args.model_name
layer = int(args.layer)
gpuid = args.gpuid
is_display = args.display
is_show = args.show

punctuation_string = string.punctuation

# Move model and tokenizer to CUDA if available
device = torch.device("cuda:"+gpuid if torch.cuda.is_available() else "cpu")

sub_ses_num = [7,7,6,8,6,7,6,7,6,6]


def main():
    global tokenizer
    global model


    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("/data/minghua/hf_output/20240116-1900/"+model_name)
    model = AutoModelForCausalLM.from_pretrained("/data/minghua/hf_output/20240116-1900/"+model_name)
    model.eval()
    model.to(device)

    save_dir = "/data/zhiang/hidden_states/" + model_name + "/total"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open txt file and get lines
    for ses in range(1, 11, 1):
        '''
        num = sub_ses_num[ses-1]
        context = []
        for sub_ses in range(1, num+1, 1):
            txt_path = "/data/zhiang/holmes_test/txt/{:>02d}_{}_t.txt".format(ses, sub_ses)
    
            for index, line in enumerate(open(txt_path)):
                context.append(line)
        '''
        context = []
        txt_path = "/data/zhiang/txt/total/{:>02d}_total.txt".format(ses)
        for index, line in enumerate(open(txt_path)):
            context.append(line)
        context_total = len(context)
        print(context_total)

        data_tensor = torch.zeros(0,4096).to(device)

        if is_show:
            print('start running... at', maket(int(time.time())))
        word_counts = 0
        with tqdm(total=context_total) as pbar:
            for i in range(context_total):
                txt = context[i]
                txt = txt.replace('-', ' ')

                s = get_layer_states_by_text(txt)

                for c in punctuation_string:
                    txt = txt.replace(c, '')
                words = txt.split()

                word_counts += len(words)
                data_tensor = torch.cat((data_tensor, s), dim=0)
                pbar.update()

        if is_show:
            print(data_tensor.shape)
            print(word_counts)
        data_array = data_tensor.data.cpu().numpy()
        save_path = save_dir + "/hs_{}.npy".format(ses)
        e_save_path = save_dir + "/e_hs_{}.npy".format(ses)
        if ses in [2,3,4,10]:
            np.save(e_save_path, data_array)
        else:
            np.save(save_path, data_array)




def get_layer_states_by_text(input_text):
    line = input_text
    line = line.strip()
    inputs = tokenizer(line, return_tensors="pt").to(device)
    if is_display:
        print('your input:')
        print(inputs)
    input_ids = inputs["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    new_tokens = []
    for token in tokens:
        new_token = token.replace('Ġ', '▁')
        new_tokens.append(new_token)
    tokens = new_tokens
    if is_display:
        print("tokens:")
        print(tokens)
    
    non_words_index = []
    words = line.split()
    words_len = len(words)
    for i in range(words_len):
        flag = True
        for j in range(len(words[i])):
            if words[i][j] not in punctuation_string:
                flag = False
        if flag:
            non_words_index.append(i)
    if is_display:
        print("non-word index:")
        print(non_words_index)
    words_num = words_len - len(non_words_index)
    untokenized_sent = tuple(words)
    if is_display:
        print("words:")
        print(untokenized_sent)
    
    mapping = match_tokenized_to_untokenized(tokens, untokenized_sent)
    if is_display:
        print("mapping:")
        print(mapping)
    
    with torch.no_grad(): # No need to compute gradients
        # Forward pass through the model with output_hidden_states=True
        outputs = model(**inputs, output_hidden_states=True)
        # Extract hidden states
        hidden_states = outputs.hidden_states
        if is_display:
            print('hidden states:')
            print(len(hidden_states))
        #print(hidden_states)
        target_hidden_layers = hidden_states[layer][0]
        if is_display:
            print(target_hidden_layers.shape)

        layer_activations = torch.zeros(words_num, target_hidden_layers.shape[1]).to(device)
        if is_display:    
            print(layer_activations.shape)
        words_count = 0
        for index in range(words_len):
            if index in non_words_index:
                continue
            if is_display:
                print(mapping[index])
            for i in mapping[index]:
                layer_activations[words_count] += target_hidden_layers[i]
            layer_activations[words_count] /= len(mapping[index])
            words_count += 1
        if is_display:
            print(words_count)
            print('result:')
            print(layer_activations)
        
        return layer_activations

def filter(sub): 
    save_dir = "/data/zhiang/hidden_states/" + model_name
    
    wd_id_droppeds = wd_id_ds[sub]

    save_path = save_dir + "/sub-{:03d}".format(sub)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tqdm(total=10) as pbar:
        for ses in range(1,11,1):
            ses_path = save_path + "/hs_{}.npy".format(ses)
            hs_path = save_dir + "/total/hs_{}.npy".format(ses)
            hs = np.load(hs_path)
            hs = np.delete(hs, wd_id_droppeds[ses-1], axis=0)
            print(hs.shape)
            np.save(ses_path, hs)
            pbar.update()


if __name__ == '__main__':
    #main()
    for i in [1,2,3]:
        filter(i)