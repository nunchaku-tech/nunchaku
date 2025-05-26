from PIL import Image, ImageChops

# 打开 RGBA 图像
img = Image.open("removal.png").convert("RGBA")

# 拆分成 R, G, B, A 四个通道
r, g, b, a = img.split()
a_inverted = ImageChops.invert(a)

# 合并 R, G, B 成 RGB 图像
rgb_img = Image.merge("RGB", (r, g, b))

# 保存 RGB 和 A 分开的图像
rgb_img.save("removal_image.png")
a_inverted.save("removal_mask.png")
