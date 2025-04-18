# llama2.rs
A rust port to run inference on the Llama 2 LLM model. Uses the `ndarray` crate to simplify some of the linear algebra.

## llama2.c

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

With the code in this repo you can train the Llama 2 LLM architecture from scratch in PyTorch, then export the weights to a binary file, and load that into one ~simple 500-line C file ([run.c](run.c)) that inferences the model. Alternatively, you can load, finetune, and inference Meta's Llama 2 (but this is still being actively fleshed out). Hence, this repo is a "fullstack" train + inference solution for Llama 2 LLM, with a focus on minimalism and simplicity. You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough. I recommend looking at the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper for inspiration.

Please note that this started recently as just a fun weekend project: I took my earlier [nanoGPT](https://github.com/karpathy/nanoGPT), tuned it to implement the Llama-2 architecture instead of GPT-2, and the meat of it was writing the C inference engine in [run.c](run.c). So the project is young and moving quickly. Hat tip to the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) for inspiring this project. I wanted something super minimal so I chose to hard-code the Llama 2 architecture, stick to fp32, and just roll one inference file of pure C with no dependencies.

## feel the magic

Let's just run a baby Llama 2 model in C. You need a model checkpoint. Download this 15M parameter model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~60MB download):

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Compile and run the C code:

```bash
make run
./run stories15M.bin
```

You'll see the text stream a sample. On my M1 MacBook Air this runs at ~110 tokens/s. See [performance](#performance) or the Makefile for compile flags that can significantly speed this up. We can also try a bit bigger 42M parameter model:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
./run stories42M.bin
```

This still runs at interactive rates and samples more coherent and diverse stories:

> Once upon a time, there was a little girl named Lily. She loved playing with her toys on top of her bed. One day, she decided to have a tea party with her stuffed animals. She poured some tea into a tiny teapot and put it on top of the teapot. Suddenly, her little brother Max came into the room and wanted to join the tea party too. Lily didn't want to share her tea and she told Max to go away. Max started to cry and Lily felt bad. She decided to yield her tea party to Max and they both shared the teapot. But then, something unexpected happened. The teapot started to shake and wiggle. Lily and Max were scared and didn't know what to do. Suddenly, the teapot started to fly towards the ceiling and landed on the top of the bed. Lily and Max were amazed and they hugged each other. They realized that sharing was much more fun than being selfish. From that day on, they always shared their tea parties and toys.

You can also prompt the model with a prefix (sadly, because this is currently done via positional arguments, you also have to specify temperature 1.0 and 256 steps, before you enter the prompt):

```bash
./run stories42M.bin 1.0 256 "One day, Lily met a Shoggoth"
```

> One day, Lily met a Shoggoth. He was very shy, but was also very generous. Lily said “Hello Shoggy! Can I be your friend?” Shoggy was happy to have a friend and said “Yes, let’s explore the universe together!” So they set off on a journey to explore the universe. As they travelled, Shoggy was happy to explain to Lily about all the wonderful things in the universe. At the end of the day, Lily and Shoggy had gathered lots of wonderful things from the universe, and they both felt very proud. They promised to explore the universe as one big pair and to never stop being generous to each other.

There is also an even better 110M param model available, see [models](#models).

## Meta's Llama 2 models

As the neural net architecture is identical, we can also inference the Llama 2 models released by Meta. Sadly there is a bit of friction here due to licensing (I can't directly upload the checkpoints, I think). So Step 1, get the Llama 2 checkpoints by following the [Meta instructions](https://github.com/facebookresearch/llama). Once we have those checkpoints, we have to convert them into the llama2.c format.
For this we need to install the python dependencies (`pip install -r requirements.txt`) and then use the `export_meta_llama_bin.py` file, e.g. for 7B model:

```bash
python export_meta_llama_bin.py path/to/llama/model/7B llama2_7b.bin
```

The export will take ~10 minutes or so and generate a 26GB file (the weights of the 7B model in float32) called `llama2_7b.bin` in the current directory. It has been [reported](https://github.com/karpathy/llama2.c/pull/85) that despite efforts, the 13B export currently doesn't work for unknown reaons (accepting PRs for fix). We can run the model as normal:

```bash
./run llama2_7b.bin
```

This ran at about 4 tokens/s compiled with [OpenMP](#OpenMP) on 96 threads on my CPU Linux box in the cloud. (On my MacBook Air M1, currently it's closer to 30 seconds per token if you just build with `make runfast`.) Example output:

> The purpose of this document is to highlight the state-of-the-art of CoO generation technologies, both recent developments and those in commercial use. The focus is on the technologies with the highest merit to become the dominating processes of the future and therefore to be technologies of interest to S&amp;T ... R&amp;D. As such, CoO generation technologies developed in Russia, Japan and Europe are described in some depth. The document starts with an introduction to cobalt oxides as complex products and a short view on cobalt as an essential material. The document continues with the discussion of the available CoO generation processes with respect to energy and capital consumption as well as to environmental damage.

base models... ¯\\_(ツ)_/¯. Since we can inference the base model, it should be possible to also inference the chat model quite easily, and have a conversation with it. And if we can find a way to run 7B more efficiently, we can start adding LoRA to our training script, and going wild with finetunes all within the repo!

## models

For the sake of examples of smaller, from-scratch models, I trained a small model series on TinyStories. All of these trained in a few hours on my training setup (4X A100 40GB GPUs). The 110M took around 24 hours. I am hosting them on huggingface hub [tinyllamas](https://huggingface.co/karpathy/tinyllamas), both in the original PyTorch .pt, and also in the llama2.c format .bin:

| model | dim | n_layers | n_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- |
| OG | 288 | 6 | 6 | 256 | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 1024 | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 1024 | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

You'll notice that the 110M model is equivalent to GPT-1 in size. Alternatively, this is also the smallest model in the GPT-2 series (`GPT-2 small`), except the max context length is only 1024 instead of 2048. The only notable changes from GPT-1/2 architecture is that Llama uses RoPE relatively positional embeddings instead of absolute/learned positional embeddings, a bit more fancy SwiGLU non-linearity in the MLP, RMSNorm instead of LayerNorm, bias=False on all Linear layers, and is optionally multiquery (but this is not yet supported in llama2.c).

## training

Let's see how we can train a baby Llama 2 from scratch using the code in this repo. First let's download and pretokenize some source dataset, e.g. I like [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) so this is the only example currently available in this repo. But it should be very easy to add datasets, see the code.

```bash
python tinystories.py download
python tinystories.py pretokenize
```

Then train our model:

```bash
python train.py
```

**brief training guide**. See the train.py script for more exotic launches and hyperparameter overrides. Here is a brief guide to how to set the parameters. Look at the table at the very end of the [Chinchilla paper](https://arxiv.org/abs/2203.15556) to get a sense of how the Transformer parameters (dim, n_layers, n_heads) grow or shrink together. Extrapolate/interpolate this pattern to get bigger or smaller transformers. Set the max context length however you wish, depending on the problem: this should be the max number of tokens that matter to predict the next token. E.g. Llama 2 uses 2048. Next, you want the _total_ batch size per update (printed by the script as "tokens per iteration will be:") to be somewhere around 100K tokens for medium-sized applications. For tiny applications it could be lower, for large training (e.g. GPTs/LLamas) it is usually ~0.5M, or even more. You get there by first maxing out the batch_size to whatever your system allows (e.g. mine was 16 in a recent run because after that my GPU runs out of memory), and then you want to increase gradient_accumulation_steps to be as high as necessary to reach the total batch size of ~100K. Finally, you want to tune your learning_rate (LR). You want this to be as high as your training allows. Very small networks can get away with a large LR (e.g. 1e-3 or even higher). Large networks need lower LRs. 3e-4 is a safe choice in most medium-sized applications, but can be too low for small networks, so try to increase it! Finally, max_iters is the length of training. Play with different settings. I mostly only ever tune these parameters and leave most of the others unchanged. Here is an example of how I trained the 110M model, which I don't think is anywhere near optimal, but looked sensible to me: dim 768, n_layers 12, n_heads 12 (so size of each head is 768 / 12 = 64 channels), seq len of 1024, batch size 16 (this is the most that fit my A100 40GB GPU), gradient_accumulation_steps = 8 was needed to get total tokens batch size to be 16 batch size * 1024 tokens in sequence * 8 grad_accum = 131,072 tokens per update. Good. Learning rate 4e-4 (probably a little too low). max_iters 200K (probably a bit too high). Dropout 0.1, as that usually helps a bit at medium size. That was it. I ran using Distributed Data Parallel (DDP) on 4 GPUs on my cloud machine, training took ~day or so.

Totally understand if you want to skip model training, for simple demo just download one of the pretrained models (see [models](#models) section), e.g.:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Once we have the model.bin file, we can inference in C. Compile the C code first:

```bash
make run
```

You can now run it simply as

```bash
./run stories15M.bin
```

Watch the tokens stream by, fun! We can also run the PyTorch inference script for a comparison. Download one of the models again from huggingface hub and point the `sample.py` script at it:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt -P out15M
mv out15M/stories15M.pt out15M/ckpt.pt # sorry the sample script current assumes this directory structure / filename...
python sample.py --out_dir=out15M
```

Which gives the same results. More detailed testing will be done in `test_all.py`. Currently you will need two files to test or sample: both the .bin file, and the .ckpt file inside a directory (see `test_all.py` for details). Sorry this is a bit janky right now, I have to think through running the tests without having to download 200MB of data. But run the tests with pytest:

```bash
$ pytest
```

## performance

There are many ways to potentially speed up this code depending on your system. Have a look at the [Makefile](Makefile), which contains a lot of notes. The `make run` command currently uses the `-O3` optimization by default, i.e.:

```bash
gcc -O3 -o run run.c -lm
```

-O3 includes optimizations that are expensive in terms of compile time and memory usage. Including vectorization, loop unrolling, and predicting branches.

To get a much better performance, try to compile with `make runfast`. This turns on the `-Ofast` flag, which includes additional optimizations that may break compliance with the C/IEEE specifications, in addition to `-O3`. See [the GCC docs](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) for more information.

Try `-march=native` to compile the program to use the architecture of the machine you're compiling on rather than a more generic CPU. This may enable additional optimizations and hardware-specific tuning such as improved vector instructions/width.

The fastest throughput I saw so far on my MacBook Air (M1) so far is with `make runfast`. 

You can also experiment with replacing `gcc` with `clang`.

### OpenMP
Big improvements can also be achieved by compiling with OpenMP, which "activates" the `#pragma omp parallel for` inside the matmul and attention, allowing the work in the loops to be split up over multiple processors.
You'll need to install the OpenMP library and the clang compiler first (e.g. `apt install clang libomp-dev` on ubuntu). I was not able to get improvements from OpenMP on my MacBook, though. Then you can compile with `make runomp`, which does:

```bash
clang -Ofast -fopenmp -march=native run.c  -lm  -o run
```

When you run inference make sure to use OpenMP flags to set the number of threads, e.g.:

```bash
OMP_NUM_THREADS=4 ./run out/model.bin
```

Depending on your system resources you may want to tweak these hyperparameters and use more threads. But more is not always better, usually this is a bit U shaped.

## platforms

On **Windows**, use `build_msvc.bat` in a Visual Studio Command Prompt to build with msvc, or you can use `make win64` to use mingw compiler toolchain from linux or windows to build the windows target. MSVC build will automatically use openmp and max threads appropriate for your CPU unless you set `OMP_NUM_THREADS` env.

On **Centos 7**, **Amazon Linux 2018** use `rungnu` Makefile target: `make rungnu` or `make runompgnu` to use openmp.

## ack

I trained the llama2.c storyteller models on a 4X A100 40GB box graciously provided by the excellent [Lambda labs](https://lambdalabs.com/service/gpu-cloud), thank you.

## discord

Figured it's possible to reuse my existing discord channel (that I use for my [zero to hero youtube series](https://karpathy.ai/zero-to-hero.html)), see #llama2c channel on [discord](https://discord.gg/3zy8kqD9Cp), for any quick questions, related discussions, etc.

## contributing

A few words on this repo and the kinds of PRs that are likely to be accepted. What is the goal of this repo? Basically I think there will be a lot of interest in training or finetuning custom micro-LLMs (think ~100M - ~1B params, but let's say up to ~10B params) across a large diversity of applications, and deploying them in edge-adjacent environments (think MCUs, phones, web browsers, laptops, etc.). I'd like this repo to be the simplest, smallest, most hackable repo to support this workflow, both training and inference. In particular, this repo is not a complex framework with a 1000 knobs controlling inscrutible code across a nested directory structure of hundreds of files. Instead, I expect most applications will wish to create a fork of this repo and hack it to their specific needs and deployment platforms.

People who care about deployment efficiency above all else should look at [llama.cpp](https://github.com/ggerganov/llama.cpp). This repo still cares about efficiency, but not at the cost of simplicity, readability or portability. Basically, I expect that a lot of people come to this repo because the training code is 2 readable .py files and the inference code is 500 lines of C. So I'd like this to continue to be a kind of simplest "reference implementation" that can be easily hacked in a separate fork into whatever downstream application people are excited about. It shouldn't be full-featured. It shouldn't take 100 different options or settings. It shouldn't be the most efficient. A few examples:

- someone re-ordered two loops to improve data locality for a small efficieny win => instant merge.
- someone added the one line "pragma omp parallel for", which allows you to compile with OpenMP and dramatically speed up the code, or acts as just a comment if you don't compile it that way => instant merge.
- bug fixes and touchups etc. => happy to merge

A few examples of PRs are that are not an excellent fit:

- adding more than several #ifdefs all over the place in code. If they are localized / few, might be okay.
- adding a lot of code that is very specific to some specific platform (e.g. MCUs, or some special version of linux or processor). These may be a better fit for forks of the project, and I am very happy to maintain a list of these forks in section below.
- adding hundreds of lines of code to run.c that are only active in specific scenarios or platforms.

If your candidate PRs have elements of these it doesn't mean they won't get merged, it just means they will make it into the gray territory. TLDR: I am eager to merge any mostly small, mostly localized, broadly applicable, clean changes that improve the efficiency and portability of the repo, while keep its hackability and readability. I appreciate all PRs seeking to help me improve the project, thank you! <3.

## notable forks

- [llama2.rs](https://github.com/gaxler/llama2.rs) by @gaxler: a Rust port of this project
- [go-llama2](https://github.com/tmc/go-llama2) by @tmc: a Go port of this project
- [llama2.go](https://github.com/nikolaydubina/llama2.go) by @nikolaydubina: a Go port of this project
- [llama2.go](https://github.com/haormj/llama2.go) by @haormj: a Go port of this project
- [llama2.go](https://github.com/saracen/llama2.go) by @saracen: a Go port of this project
- [llama2.c-android](https://github.com/Manuel030/llama2.c-android): by @Manuel030: adds Android binaries of this project
- [llama2.cpp](https://github.com/leloykun/llama2.cpp) by @leloykun: a C++ port of this project
- [llama2.js](https://github.com/epicure/llama2.js) by @epicure: a JavaScript port of this project
- [llama2.zig](https://github.com/cgbur/llama2.zig) by @cgbur: A Zig port of this project

## unsorted todos

- support Llama 2 7B Chat model and tune run.c to Chat UI/UX
- speed up 7B Llama 2 models sufficiently to work at interactive rates on Apple Silicon MacBooks
- possibly include emscripten / web backend (as seen in @gg PR)
- currently the project only runs in fp32, how easy would it be to different precisions?
- look into quantization and what would be involved
- todo multiquery support? doesn't seem as useful for smaller models that run on CPU (?)
- todo support inferencing beyond max_seq_len steps, have to think through the kv cache
- why is MFU so low (~10%) on my A100 40GB for training?
- weird errors with torch.compile and wandb when using DDP
- (LoRA) finetuning of Llama 2 models
- make more better tests to decrease yolo

## License

MIT
