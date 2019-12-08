import torch
import bootstrap.lib.utils as utils
import bootstrap.engines as engines
import bootstrap.models as models
import bootstrap.datasets as datasets
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import init_logs_options_files
from block.datasets.vqa_utils import tokenize_mcb
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Options()
    #print("seed: {}".format(Options()['misc']['seed']))
    utils.set_random_seed(Options()['misc']['seed'])
    init_logs_options_files(Options()['exp']['dir'], Options()['exp']['resume'])

    engine = engines.factory()
    #print("engine: {}".format(engine))
    engine.dataset = datasets.factory(engine)
    engine.model = models.factory(engine)
    #print("engine.model: {}".format(engine.model))
    engine.model.eval()
    engine.resume()

    #img_emb_module = load_img_emb_module().cuda() #engine.model.network.image_embedding

    # inputs
    # img = torch.randn(1, 3, 224, 244).cuda()
    # question = 'What is the color of the white horse?'
    # question_words = tokenize_mcb(question)
    # question_wids = [engine.dataset['eval'].word_to_wid[word] for word in question_words]

    # item = {
    #     'question': torch.LongTensor(question_wids),
    #     'lengths': torch.LongTensor([len(question_wids)]),
    # }
    #print("engine.dataset['eval']: {}".format(engine.dataset['eval']))
    #batch = engine.dataset['eval'].items_tf()([item])
    #batch = engine.dataset['eval'].add_rcnn_to_item()([item])
    print("engine.dataset: {}".format(engine.dataset))
    batch, question_str, image_name = engine.dataset['eval'].__getitem__(4)
    batch['question'] = torch.unsqueeze(batch['question'], 0)
    batch['lengths'] = torch.unsqueeze(batch['lengths'], 0)
    batch['visual'] = torch.unsqueeze(batch['visual'], 0)
    #print("visual: {}".format(batch['visual'].shape))
    #batch = engine.model.prepare_batch(batch)
    #print("batch: {}".format(batch))

    with torch.no_grad():
        #batch['visual'] = img_emb_module(img)  # 1 x nb_regions x 2048 (nb_regions=36 in our case)
        out = engine.model.network.forward(batch)
        #print("out1: {}".format(out))
        #print("out1['logits']: {}".format(out['logits'].shape))
        out = engine.model.network.process_answer(out)
        print("Question: {}".format(question_str))
        print("Answer: {}".format(out['answers']))
        raw_image = cv2.imread("data/vqa/coco/raw/val2014/" + image_name)
        plt.imshow(raw_image)
        plt.savefig('results/raw_image.png')

    Logger()(out['answers'])