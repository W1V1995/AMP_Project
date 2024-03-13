import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel,GPT2Tokenizer
import random
import torch
import numpy as np

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]


def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    generate = [] + context
    with torch.no_grad():
        for i in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generate.append(next_token.item())
            prev = next_token.view(1, 1)
    return generate



def generate(n_ctx, model, context, length, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu',
             is_fast_pattern=False):

    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='GPU or not')
    parser.add_argument('--random_seed', default=666, type=str, required=False, help='random seed')
    parser.add_argument("--ntokens", type=str, help="Length of tokens (e.g., 8-15).")
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='batch size')
    parser.add_argument('--nsamples', default=100, type=int, required=False, help='Number of samples')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='topK')
    parser.add_argument('--topp', default=0, type=float, required=False, help='topP')
    parser.add_argument('--model_path', default='/AMP_models/AmpGenix/', type=str, required=False, help='model path')
    parser.add_argument('--prefix', default='S', type=str, required=False, help='prefix')
    parser.add_argument('--fast_pattern', action='store_true', help='faster generation')
    parser.add_argument('--save_samples', action='store_true', help='samples saved')
    parser.add_argument('--save_samples_path', default='/Data/samples_saved/prefix_S/', type=str, required=False, help='samples saved path')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='repetition penalty')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    batch_size = args.batch_size
    nsamples = args.nsamples
    temperature = args.temperature
    prefix = args.prefix
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    random_seed = args.random_seed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx


    ntokens = args.ntokens.split("-")
    start_size = int(ntokens[0])
    end_size = int(ntokens[1])
    window_sizes = list(range(start_size, end_size + 1))

    for length in window_sizes:
        if length == -1:
            length = model.config.n_ctx
        if args.save_samples:
            if not os.path.exists(args.save_samples_path):
                os.makedirs(args.save_samples_path)
            samples_file = open(args.save_samples_path + f'/samples_{prefix}_{length}.csv', 'w', encoding='utf8')
        while True:
            raw_text = args.prefix
            context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
            generated = 0
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            for _ in range(nsamples // batch_size):
                out = generate(
                    n_ctx=n_ctx,
                    model=model,
                    context=context_tokens,
                    length=length,
                    is_fast_pattern=args.fast_pattern, tokenizer=tokenizer,
                    temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device
                )
                for i in range(batch_size):
                    generated += 1
                    text = tokenizer.convert_ids_to_tokens(out)
                    #if '<|endoftext|>' in text:
                        
                    #    break
                    for i, item in enumerate(text):
                        if item == '[MASK]':
                            text[i] = ''
                        elif item == '[CLS]':
                            text[i] = '\n\n'
                        elif item == '[SEP]':
                            text[i] = '\n'
                    info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
                    print(info)
                    text = ''.join(text).replace('##', '').strip()
                    print(text)
                    if args.save_samples:
                        #samples_file.write(info)
                        samples_file.write(text)
                        samples_file.write('\n')
                        #samples_file.write('=' * 90)
                        #samples_file.write('\n' * 2)
            print("=" * 80)
            if generated == nsamples:
                # close file when finish writing.
                if args.save_samples:
                    samples_file.close()
                break


if __name__ == '__main__':
    main()
