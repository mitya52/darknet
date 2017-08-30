//#define USE_GPU // dirty hack...
extern "C" {
#ifndef USE_GPU
#include "include/darknet.h"
#else
#include "include/darknet_gpu.h"
#endif
}

void cuda_set_device(int);

float thresh = 0.24;
float hier_thresh = 0.5;

void free_ptrs(void **ptrs, int n)
{
    int i;
    for(i = 0; i < n; ++i) free(ptrs[i]);
    free(ptrs);
}

// usage executable cfg/yolo.cfg data/dog.jpg data/coco.names
int main(int argc, char* argv[])
{
    char weightfile[] = "yolo.weights";
    char outfile[] = "output";

    char **names = get_labels(argv[3]);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(argv[1]);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    double time;
    float nms=.3;

#ifdef USE_GPU
    cuda_set_device(0);
#endif

    image im = load_image_color(argv[2], 0, 0);
    image sized = letterbox_image(im, net.w, net.h);
    layer l = net.layers[net.n-1];

    box* boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
    float** probs = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
    for (int j = 0; j < l.w*l.h*l.n; ++j)
    	probs[j] = (float*)calloc(l.classes + 1, sizeof(float*));
    float** masks = 0;
    if (l.coords > 4) {
        masks = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
        for (int j = 0; j < l.w*l.h*l.n; ++j)
        	masks[j] = (float*)calloc(l.coords-4, sizeof(float*));
    }

    float *X = sized.data;
    network_predict(net, X);
    get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
    if (nms)
    	do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
    if (outfile) {
        save_image(im, outfile);
    }

     free_image(im);
     free_image(sized);
     free(boxes);
     free_ptrs((void **)probs, l.w*l.h*l.n);
}