from PIL import Image

def crop_to_multiple(img, multiple=8):
    w, h = img.size
    w = (w // multiple) * multiple
    h = (h // multiple) * multiple
    return img.crop((0, 0, w, h))

def make_lr(hr_img, scale=4):
    # crop so HR is compatible with SD and scaling
    hr_img = crop_to_multiple(hr_img, multiple=8 * scale)

    w, h = hr_img.size
    lr = hr_img.resize(
        (w // scale, h // scale),
        resample=Image.BICUBIC
    )
    return lr, hr_img

