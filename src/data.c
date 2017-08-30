#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char **replace_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip = rand()%2;
        if (flip) flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        /*
        show_image(im, "orig");
        show_image(crop, "crop");
        cvWaitKey(0);
        */
        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }
    return X;
}


box_label *read_boxes(char *filename, int *n)
{
    box_label *boxes = calloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = realloc(boxes, (count+1)*sizeof(box_label));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 30; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .005 || h < .005) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

void load_rle(image im, int *rle, int n)
{
    int count = 0;
    int curr = 0;
    int i,j;
    for(i = 0; i < n; ++i){
        for(j = 0; j < rle[i]; ++j){
            im.data[count++] = curr;
        }
        curr = 1 - curr;
    }
    for(; count < im.h*im.w*im.c; ++count){
        im.data[count] = curr;
    }
}

void or_image(image src, image dest, int c)
{
    int i;
    for(i = 0; i < src.w*src.h; ++i){
        if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
    }
}

void exclusive_image(image src)
{
    int k, j, i;
    int s = src.w*src.h;
    for(k = 0; k < src.c-1; ++k){
        for(i = 0; i < s; ++i){
            if (src.data[k*s + i]){
                for(j = k+1; j < src.c; ++j){
                    src.data[j*s + i] = 0;
                }
            }
        }
    }
}

box bound_image(image im)
{
    int x,y;
    int minx = im.w;
    int miny = im.h;
    int maxx = 0;
    int maxy = 0;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            if(im.data[y*im.w + x]){
                minx = (x < minx) ? x : minx;
                miny = (y < miny) ? y : miny;
                maxx = (x > maxx) ? x : maxx;
                maxy = (y > maxy) ? y : maxy;
            }
        }
    }
    box b = {minx, miny, maxx-minx + 1, maxy-miny + 1};
    //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
    return b;
}

void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if(flip) flip_image(sized);
        box b = bound_image(sized);
        if(b.w > 0){
            image crop = crop_image(sized, b.x, b.y, b.w, b.h);
            image mask = resize_image(crop, mw, mh);
            truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.)/sized.w;
            truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.)/sized.h;
            truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
            truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;
            int j;
            for(j = 0; j < mw*mh; ++j){
                truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
            }
            truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = id;
            free_image(crop);
            free_image(mask);
            ++i;
        }
        free_image(sized);
        free(rle);
    }
    fclose(file);
    free_image(part);
}


void fill_truth_detection(char *path, int num_boxes, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    find_replace(path, "images", "labels", labelpath);
    find_replace(labelpath, "JPEGImages", "labels", labelpath);

    find_replace(labelpath, "raw", "labels", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001)) continue;

        truth[i*5+0] = x;
        truth[i*5+1] = y;
        truth[i*5+2] = w;
        truth[i*5+3] = h;
        truth[i*5+4] = id;
    }
    free(boxes);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if(count != 1 && (k != 1 || count != 0)) printf("Too many or too few labels: %d, %s\n", count, path);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_regression_labels_paths(char **paths, int n)
{
    matrix y = make_matrix(n, 1);
    int i;
    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(paths[i], "images", "targets", labelpath);
        find_replace(labelpath, "JPEGImages", "targets", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);

        FILE *file = fopen(labelpath, "r");
        fscanf(file, "%f", &(y.vals[i][0]));
        fclose(file);
    }
    return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "imgs", "labels", label);
        find_replace(label, "_iconl.jpeg", ".txt", label);
        FILE *file = fopen(label, "r");
        if(!file){
            find_replace(label, "labels", "labels2", label);
            file = fopen(label, "r");
            if(!file) continue;
        }
        ++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    printf("%d/%d\n", count, n);
    return y;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

image get_segmentation_image(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes+1);
    int i;
    for(i = 0; i < w*h; ++i){
        mask.data[w*h*classes + i] = 1;
    }
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        for(i = 0; i < w*h; ++i){
            if(part.data[i]) mask.data[w*h*classes + i] = 0;
        }
        free(rle);
    }
    //exclusive_image(mask);
    fclose(file);
    free_image(part);
    return mask;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = rand()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*30;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = rand()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d = {0};
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

