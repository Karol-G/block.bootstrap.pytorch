import torch
import bootstrap.lib.utils as utils
import bootstrap.engines as engines
import bootstrap.models as models
import bootstrap.datasets as datasets
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
from bootstrap.run import init_logs_options_files
from block.datasets.vqa_utils import tokenize_mcb

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
    img = torch.randn(1, 3, 224, 244).cuda()
    question = 'What is the color of the white horse?'
    question_words = tokenize_mcb(question)
    question_wids = [engine.dataset['eval'].word_to_wid[word] for word in question_words]

    item = {
        'question': torch.LongTensor(question_wids),
        'lengths': torch.LongTensor([len(question_wids)]),
    }
    #print("engine.dataset['eval']: {}".format(engine.dataset['eval']))
    #batch = engine.dataset['eval'].items_tf()([item])
    batch = engine.dataset['eval'].add_rcnn_to_item()([item])
    batch = engine.model.prepare_batch(batch)

    with torch.no_grad():
        batch['visual'] = img_emb_module(img)  # 1 x nb_regions x 2048 (nb_regions=36 in our case)
        out = engine.model.network(batch)
        out = engine.model.network.process_answers(out)

    Logger()(out['answers'][0])  # white