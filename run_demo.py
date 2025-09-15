import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat


from model import model

#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='model')
parser.add_argument('--weight_name', '-w', type=str, default='model')
parser.add_argument('--file_name', '-f', type=str, default='final.pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=1)
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=480)
parser.add_argument('--img_width', '-iw', type=int, default=640)
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--n_class', '-nc', type=int, default=7)
parser.add_argument('--data_dir', '-dr', type=str, default='E:\hxy\Datasets')
parser.add_argument('--model_dir', '-wd', type=str, default='E:\hxy\Datasets\Weights')
args = parser.parse_args()
#############################################################################################



if __name__ == '__main__':


    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." %(model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    conf_total = np.zeros((args.n_class, args.n_class))
    model = eval(args.model_name)(n_class=args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location = lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1
    test_dataset  = MF_dataset(data_dir=args.data_dir, split=args.dataset_split, input_h=args.img_height, input_w=args.img_width)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    ave_time_cost = 0.0

    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            thermal = images[:,3:]
            torch.cuda.synchronize()
            start_time = time.time()

            Tfeatures,logits,Thint = model(images)
            torch.cuda.synchronize()
            end_time = time.time()


            memory_allocated = torch.cuda.memory_allocated(args.gpu) / (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved(args.gpu) / (1024 * 1024)

            if it>=5:
                ave_time_cost += (end_time-start_time)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6])
            conf_total += conf

            visualize(image_name=names, predictions=logits.argmax(1), weight_name=args.weight_name)

            print(f"Frame {it + 1}/{len(test_loader)}:")
            print(f"    Time cost: {(end_time - start_time) * 1000:.2f} ms")
            print(f"    Memory Allocated: {memory_allocated:.2f} MB")
            print(f"    Memory Reserved: {memory_reserved:.2f} MB")
            # save demo images
            visualize(image_name=names, predictions=logits.argmax(1), weight_name=args.weight_name)
            print("%s, %s, frame %d/%d, %s, time cost: %.2f ms, demo result saved."
                  %(args.model_name, args.weight_name, it+1, len(test_loader), names, (end_time-start_time)*1000))

    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
    conf_total_matfile = os.path.join("./runs", 'conf_'+args.weight_name+'.mat')
    savemat(conf_total_matfile,  {'conf': conf_total}) # 'conf' is the variable name when loaded in Matlab

    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' %(args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' %(args.img_height, args.img_width))
    print('* the weight name: %s' %args.weight_name)
    print('* the file name: %s' %args.file_name)
    print("* iou per class: \n     unlabeled: %.6f, reban: %.6f, yinying: %.6f, yiwu: %.6f, niaofen: %.6f, duanlu: %.6f, tree: %.6f" \
        % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3], iou_per_class[4], iou_per_class[5], iou_per_class[6]))
    print("* recall per class: \n    unlabeled: %.6f, reban: %.6f, yinying: %.6f, yiwu: %.6f, niaofen: %.6f, duanlu: %.6f, tree: %.6f" \
        % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3], recall_per_class[4], recall_per_class[5], recall_per_class[6]))
    print("* pre per class: \n    unlabeled: %.6f, reban: %.6f, yinying: %.6f, yiwu: %.6f, niaofen: %.6f, duanlu: %.6f, tree: %.6f" \
        % (precision_per_class[0], precision_per_class[1], precision_per_class[2], precision_per_class[3], precision_per_class[4], precision_per_class[5], precision_per_class[6]))
    print("\n* average values (np.mean(x)): \n iou: %.6f, recall: %.6f" \
          %(iou_per_class.mean(),recall_per_class.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n iou: %.6f, recall: %.6f" \
          %(np.mean(np.nan_to_num(iou_per_class)), np.mean(np.nan_to_num(recall_per_class))))
    print('\n* the average time cost per frame (with batch size %d): %.2f ms, namely, the inference speed is %.2f fps' %(batch_size, ave_time_cost*1000/(len(test_loader)-5), 1.0/(ave_time_cost/(len(test_loader)-5))))
    print('\n###########################################################################')