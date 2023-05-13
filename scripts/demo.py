from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt

if __name__ == "__main__":
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

    im_fname = 'resources/dog_bike_car.jpg'
    # NOTE 如果更换了其他模型，这个图片加载方法也应该相应更改
    x, orig_img = data.transforms.presets.yolo.load_test(im_fname)

    class_IDs, scores, bounding_boxs = net(x)

    ax = utils.viz.plot_bbox(orig_img, bounding_boxs[0], scores[0],
                            class_IDs[0], class_names=net.classes)

    plt.savefig("demo_result.jpg")
