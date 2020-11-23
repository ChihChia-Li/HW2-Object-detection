from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import json


def main():
    mydict = {'bbox':[],'score':[],'label':[]}
    result_list = []
    threshold = 0
    num = 1

    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--root', type=str, default=None)
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test all images
    for idx in range(1, 13069):
        result = inference_detector(model, args.root + str(idx) + '.png')
        result =  [r.tolist()  for r in result]
        for arr in result:
            if(len(arr) > 0):
                for k in arr:
                    if(float(k[-1] ) >=  threshold):
                        mydict['bbox'].append(tuple((round(k[1]),  round(k[0]),  round(k[3]),  round(k[2]))))
                        mydict['score'].append(float(k[-1]))
                        mydict['label'].append(int(num))
            num += 1
        result_list.append(mydict)
        num=1
        mydict = {'bbox':[], 'score':[], 'label':[]}

    with open('309553007.json', 'w') as fp:
        json.dump(result_list,  fp)

    # show the results
# show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
