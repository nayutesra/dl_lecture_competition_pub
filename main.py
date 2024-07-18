import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed):
    """
    乱数のシードを設定する関数。

    Parameters
    ----------
    seed : int
        乱数生成に使用するシード値
    """
    random.seed(seed)  # randomモジュールのシードを設定
    np.random.seed(seed)  # numpyのシードを設定
    torch.manual_seed(seed)  # torchのシードを設定
    torch.cuda.manual_seed(seed)  # CUDAのシードを設定
    torch.cuda.manual_seed_all(seed)  # すべてのGPUでCUDAのシードを設定
    torch.backends.cudnn.deterministic = True  # 再現性のためにCUDNNを決定論的に設定
    torch.backends.cudnn.benchmark = False  # 再現性のためにCUDNNベンチマークを無効化

def process_text(text):
    """
    テキストを前処理する関数。

    Parameters
    ----------
    text : str
        前処理するテキスト

    Returns
    -------
    str
        前処理されたテキスト
    """
    text = text.lower()  # 小文字に変換

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():  # 数詞を数字に置換
        text = text.replace(word, digit)

    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)  # 小数点のピリオドを削除

    text = re.sub(r'\b(a|an|the)\b', '', text)  # 冠詞を削除

    # 短縮形の追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():  # 短縮形を正しい形に置換
        text = text.replace(contraction, correct)

    text = re.sub(r"[^\w\s':]", ' ', text)  # 句読点をスペースに変換

    text = re.sub(r'\s+,', ',', text)  # 連続するスペースをカンマに置換

    text = re.sub(r'\s+', ' ', text).strip()  # 連続するスペースを1つに変換し、前後のスペースを削除

    return text  # 前処理されたテキストを返す

def preprocess_question(question):
    """
    質問を前処理して単語に分割する関数。

    Parameters
    ----------
    question : str
        前処理する質問

    Returns
    -------
    list of str
        前処理された単語のリスト
    """
    question = process_text(question)  # テキストを前処理
    return question.split(" ")  # 単語に分割してリストとして返す

class VQADataset(torch.utils.data.Dataset):
    """
    VQA (Visual Question Answering) 用のデータセットクラス。

    Parameters
    ----------
    df_path : str
        データセットのJSONファイルのパス
    image_dir : str
        画像が保存されているディレクトリ
    transform : torchvision.transforms.Compose, optional
        画像に適用する変換 (デフォルトはNone)
    answer : bool, optional
        回答を含むかどうか (デフォルトはTrue)
    """

    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の変換を設定
        self.image_dir = image_dir  # 画像のディレクトリを設定
        self.df = pandas.read_json(df_path)  # データフレームを読み込む
        self.answer = answer  # 回答を含むかどうかを設定

        self.question2idx = {}  # 質問の単語からインデックスへの辞書
        self.answer2idx = {}  # 回答の単語からインデックスへの辞書
        self.idx2question = {}  # インデックスから質問の単語への辞書
        self.idx2answer = {}  # インデックスから回答の単語への辞書

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            words = preprocess_question(question)  # 質問を前処理して単語に分割
            for word in words:  # 各単語に対して
                if word not in self.question2idx:  # 辞書に単語がなければ追加
                    self.question2idx[word] = len(self.question2idx)
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # インデックスから質問の単語への辞書を作成

        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])  # 回答を前処理
                    if word not in self.answer2idx:  # 辞書に単語がなければ追加
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # インデックスから回答の単語への辞書を作成

    def update_dict(self, dataset):
        """
        他のデータセットから辞書を更新する関数。

        Parameters
        ----------
        dataset : VQADataset
            辞書を更新するためのデータセット
        """
        self.question2idx = dataset.question2idx  # 質問の辞書を更新
        self.answer2idx = dataset.answer2idx  # 回答の辞書を更新
        self.idx2question = dataset.idx2question  # インデックスから質問の辞書を更新
        self.idx2answer = dataset.idx2answer  # インデックスから回答の辞書を更新

    def __getitem__(self, idx):
        """
        データセットからアイテムを取得する関数。

        Parameters
        ----------
        idx : int
            取得するアイテムのインデックス

        Returns
        -------
        tuple
            画像、質問のテンソル、およびオプションで回答とモード回答のインデックスのタプル
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert("RGB")  # 画像を読み込み
        if self.transform:
            image = self.transform(image)  # 画像に変換を適用
        question = torch.tensor([self.question2idx.get(word, len(self.question2idx)) for word in preprocess_question(self.df["question"][idx])])  # 質問をテンソルに変換

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]  # 回答をテンソルに変換
            mode_answer_idx = mode(answers)  # モード回答のインデックスを取得

            return image, question, torch.tensor(answers), int(mode_answer_idx)  # 画像、質問、回答、モード回答のインデックスを返す

        else:
            return image, question  # 画像と質問を返す

    def __len__(self):
        """
        データセットのアイテム数を取得する関数。

        Returns
        -------
        int
            データセットのアイテム数
        """
        return len(self.df)  # データフレームの長さを返す

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    """
    VQA (Visual Question Answering) の評価指標を計算する関数。

    Parameters
    ----------
    batch_pred : torch.Tensor
        モデルの予測
    batch_answers : torch.Tensor
        正解の回答

    Returns
    -------
    float
        計算された精度
    """
    total_acc = 0.  # 総精度を初期化

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.  # 各サンプルの精度を初期化
        for i in range(len(answers)):  # 各回答に対して
            num_match = 0  # 一致する回答の数を初期化
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:  # 予測が回答と一致する場合
                    num_match += 1
            acc += min(num_match / 3, 1)  # 精度を計算して追加
        total_acc += acc / 10  # サンプルごとの精度を総精度に追加

    return total_acc / len(batch_pred)  # 平均精度を返す

class BasicBlock(nn.Module):
    """
    ResNetアーキテクチャの基本ブロック。

    Parameters
    ----------
    in_channels : int
        入力チャンネル数
    out_channels : int
        出力チャンネル数
    stride : int, optional
        畳み込みのストライド (デフォルトは1)
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 畳み込み層1
        self.bn1 = nn.BatchNorm2d(out_channels)  # バッチ正規化1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)  # 畳み込み層2
        self.bn2 = nn.BatchNorm2d(out_channels)  # バッチ正規化2
        self.relu = nn.ReLU(inplace=True)  # ReLU活性化関数

        self.shortcut = nn.Sequential()  # ショートカット接続
        if stride != 1 or in_channels != out_channels:  # 入力と出力のサイズが異なる場合
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),  # 畳み込み層
                nn.BatchNorm2d(out_channels)  # バッチ正規化
            )

    def forward(self, x):
        """
        基本ブロックの順伝播処理。

        Parameters
        ----------
        x : torch.Tensor
            入力テンソル

        Returns
        -------
        torch.Tensor
            出力テンソル
        """
        residual = x  # 残差接続のための入力を保存
        out = self.relu(self.bn1(self.conv1(x)))  # 畳み込み1 -> バッチ正規化1 -> ReLU
        out = self.bn2(self.conv2(out))  # 畳み込み2 -> バッチ正規化2

        out += self.shortcut(residual)  # ショートカット接続を追加
        out = self.relu(out)  # ReLU活性化

        return out  # 出力テンソルを返す

class BottleneckBlock(nn.Module):
    """
    ResNetアーキテクチャのボトルネックブロック。

    Parameters
    ----------
    in_channels : int
        入力チャンネル数
    out_channels : int
        出力チャンネル数
    stride : int, optional
        畳み込みのストライド (デフォルトは1)
    """
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)  # 畳み込み層1
        self.bn1 = nn.BatchNorm2d(out_channels)  # バッチ正規化1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)  # 畳み込み層2
        self.bn2 = nn.BatchNorm2d(out_channels)  # バッチ正規化2
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)  # 畳み込み層3
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  # バッチ正規化3
        self.relu = nn.ReLU(inplace=True)  # ReLU活性化関数

        self.shortcut = nn.Sequential()  # ショートカット接続
        if stride != 1 or in_channels != out_channels * self.expansion:  # 入力と出力のサイズが異なる場合
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),  # 畳み込み層
                nn.BatchNorm2d(out_channels * self.expansion)  # バッチ正規化
            )

    def forward(self, x):
        """
        ボトルネックブロックの順伝播処理。

        Parameters
        ----------
        x : torch.Tensor
            入力テンソル

        Returns
        -------
        torch.Tensor
            出力テンソル
        """
        residual = x  # 残差接続のための入力を保存
        out = self.relu(self.bn1(self.conv1(x)))  # 畳み込み1 -> バッチ正規化1 -> ReLU
        out = self.relu(self.bn2(self.conv2(out)))  # 畳み込み2 -> バッチ正規化2 -> ReLU
        out = self.bn3(self.conv3(out))  # 畳み込み3 -> バッチ正規化3

        out += self.shortcut(residual)  # ショートカット接続を追加
        out = self.relu(out)  # ReLU活性化

        return out  # 出力テンソルを返す

class ResNet(nn.Module):
    """
    ResNetアーキテクチャ。

    Parameters
    ----------
    block : nn.Module
        ブロックの種類 (BasicBlockまたはBottleneckBlock)
    layers : list of int
        各層のブロック数
    """

    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64  # 入力チャンネル数を初期化

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 畳み込み層1
        self.bn1 = nn.BatchNorm2d(64)  # バッチ正規化1
        self.relu = nn.ReLU(inplace=True)  # ReLU活性化関数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大プーリング層

        self.layer1 = self._make_layer(block, layers[0], 64)  # レイヤー1を作成
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)  # レイヤー2を作成
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)  # レイヤー3を作成
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)  # レイヤー4を作成

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自適応平均プーリング層
        self.fc = nn.Linear(512 * block.expansion, 512)  # 全結合層

    def _make_layer(self, block, blocks, out_channels, stride=1):
        """
        ResNetレイヤーを作成する関数。

        Parameters
        ----------
        block : nn.Module
            ブロックの種類 (BasicBlockまたはBottleneckBlock)
        blocks : int
            レイヤー内のブロック数
        out_channels : int
            出力チャンネル数
        stride : int, optional
            畳み込みのストライド (デフォルトは1)

        Returns
        -------
        nn.Sequential
            作成されたレイヤー
        """
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))  # 最初のブロックを追加
        self.in_channels = out_channels * block.expansion  # 入力チャンネル数を更新
        for _ in range(1, blocks):  # 残りのブロックを追加
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)  # レイヤーを順次実行するシーケンスとして返す

    def forward(self, x):
        """
        ResNetモデルの順伝播処理。

        Parameters
        ----------
        x : torch.Tensor
            入力テンソル

        Returns
        -------
        torch.Tensor
            出力テンソル
        """
        x = self.relu(self.bn1(self.conv1(x)))  # 畳み込み1 -> バッチ正規化1 -> ReLU
        x = self.maxpool(x)  # 最大プーリング

        x = self.layer1(x)  # レイヤー1
        x = self.layer2(x)  # レイヤー2
        x = self.layer3(x)  # レイヤー3
        x = self.layer4(x)  # レイヤー4

        x = self.avgpool(x)  # 自適応平均プーリング
        x = x.view(x.size(0), -1)  # 平坦化
        x = self.fc(x)  # 全結合層

        return x  # 出力テンソルを返す

def ResNet18():
    """
    ResNet-18モデルを作成する関数。

    Returns
    -------
    ResNet
        ResNet-18モデル
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    """
    ResNet-50モデルを作成する関数。

    Returns
    -------
    ResNet
        ResNet-50モデル
    """
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

class VQAModel(nn.Module):
    """
    ResNetとLSTMを用いたVQA (Visual Question Answering) モデル。

    Parameters
    ----------
    vocab_size : int
        語彙のサイズ
    n_answer : int
        回答の数
    """

    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()  # ResNet-18を使用
        self.text_embedding = nn.Embedding(vocab_size + 1, 512, padding_idx=vocab_size)  # テキスト埋め込み層
        self.lstm = nn.LSTM(512, 512, batch_first=True)  # LSTM層

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),  # 全結合層1
            nn.ReLU(inplace=True),  # ReLU活性化
            nn.Linear(512, n_answer)  # 全結合層2
        )

    def forward(self, image, question):
        """
        VQAモデルの順伝播処理。

        Parameters
        ----------
        image : torch.Tensor
            入力画像テンソル
        question : torch.Tensor
            入力質問テンソル

        Returns
        -------
        torch.Tensor
            予測された回答のテンソル
        """
        image_feature = self.resnet(image)  # 画像特徴量を取得
        question_embedding = self.text_embedding(question)  # 質問を埋め込み
        _, (question_feature, _) = self.lstm(question_embedding)  # 質問特徴量を取得
        question_feature = question_feature.squeeze(0)  # 次元を削減

        x = torch.cat([image_feature, question_feature], dim=1)  # 画像特徴量と質問特徴量を結合
        x = self.fc(x)  # 全結合層を通す

        return x  # 出力テンソルを返す

def train(model, dataloader, optimizer, criterion, device):
    """
    VQAモデルを訓練する関数。

    Parameters
    ----------
    model : nn.Module
        訓練するVQAモデル
    dataloader : torch.utils.data.DataLoader
        訓練データのデータローダー
    optimizer : torch.optim.Optimizer
        訓練に使用するオプティマイザ
    criterion : nn.Module
        損失関数
    device : str
        モデルを訓練するデバイス ('cuda' または 'cpu')

    Returns
    -------
    tuple
        訓練損失、精度、シンプル精度、訓練時間
    """
    model.train()  # モデルを訓練モードに設定

    total_loss = 0  # 総損失を初期化
    total_acc = 0  # 総精度を初期化
    simple_acc = 0  # シンプル精度を初期化

    start = time.time()  # 訓練開始時間を記録
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)  # データをデバイスに移動

        pred = model(image, question)  # モデルの予測を取得
        loss = criterion(pred, mode_answer.squeeze())  # 損失を計算

        optimizer.zero_grad()  # 勾配を初期化
        loss.backward()  # 勾配を計算
        optimizer.step()  # オプティマイザをステップ実行

        total_loss += loss.item()  # 総損失を更新
        total_acc += VQA_criterion(pred.argmax(1), answers)  # 総精度を更新
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # シンプル精度を更新

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start  # 訓練結果を返す

def eval(model, dataloader, criterion, device):
    """
    VQAモデルを評価する関数。

    Parameters
    ----------
    model : nn.Module
        評価するVQAモデル
    dataloader : torch.utils.data.DataLoader
        評価データのデータローダー
    criterion : nn.Module
        損失関数
    device : str
        モデルを評価するデバイス ('cuda' または 'cpu')

    Returns
    -------
    tuple
        評価損失、精度、シンプル精度、評価時間
    """
    model.eval()  # モデルを評価モードに設定

    total_loss = 0  # 総損失を初期化
    total_acc = 0  # 総精度を初期化
    simple_acc = 0  # シンプル精度を初期化

    start = time.time()  # 評価開始時間を記録
    for image, question, answers, mode_answer in dataloader:
        image, question, answers, mode_answer = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device)  # データをデバイスに移動

        pred = model(image, question)  # モデルの予測を取得
        loss = criterion(pred, mode_answer.squeeze())  # 損失を計算

        total_loss += loss.item()  # 総損失を更新
        total_acc += VQA_criterion(pred.argmax(1), answers)  # 総精度を更新
        simple_acc += (pred.argmax(1) == mode_answer).mean().item()  # シンプル精度を更新

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start  # 評価結果を返す

def collate_fn(batch):
    """
    DataLoaderのためのカスタムコレート関数。

    Parameters
    ----------
    batch : list
        データセットからのサンプルリスト

    Returns
    -------
    tuple
        バッチ処理された画像と質問、およびオプションで回答とモード回答
    """
    if len(batch[0]) == 4:  # 訓練/検証データ
        images, questions, answers, mode_answers = zip(*batch)  # データをアンパック
        images = torch.stack(images)  # 画像をバッチ処理
        questions = pad_sequence(questions, batch_first=True, padding_value=0)  # 質問をバッチ処理
        answers = torch.stack(answers)  # 回答をバッチ処理
        mode_answers = torch.tensor(mode_answers)  # モード回答をバッチ処理
        return images, questions, answers, mode_answers  # バッチ処理されたデータを返す
    else:  # テストデータ
        images, questions = zip(*batch)  # データをアンパック
        images = torch.stack(images)  # 画像をバッチ処理
        questions = pad_sequence(questions, batch_first=True, padding_value=0)  # 質問をバッチ処理
        return images, questions  # バッチ処理されたデータを返す

def main():
    """
    VQAモデルのセットアップと訓練を行うメイン関数。
    """
    set_seed(42)  # シードを設定
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 使用するデバイスを設定
    print(f"device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 画像をリサイズ
        transforms.ToTensor(),  # 画像をテンソルに変換
        transforms.RandomHorizontalFlip(),  # ランダムに水平反転
        transforms.RandomRotation(10),  # ランダムに回転
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 色調補正
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # ランダム消去
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)  # 訓練データセット
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transforms.Compose([
        transforms.Resize((224, 224)),  # 画像をリサイズ
        transforms.ToTensor()  # 画像をテンソルに変換
    ]), answer=False)  # テストデータセット
    test_dataset.update_dict(train_dataset)  # テストデータセットの辞書を更新

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)  # 訓練データローダー
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)  # テストデータローダー

    model = VQAModel(vocab_size=len(train_dataset.question2idx), n_answer=len(train_dataset.answer2idx)).to(device)  # モデルを作成してデバイスに移動

    num_epoch = 10  # エポック数を設定
    criterion = nn.CrossEntropyLoss()  # 損失関数を設定
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # オプティマイザを設定

    for epoch in range(num_epoch):  # 各エポックで
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)  # モデルを訓練
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")

    model.eval()  # モデルを評価モードに設定
    submission = []
    for batch in test_loader:
        image, question = batch  # テストデータを取得
        image, question = image.to(device), question.to(device)  # デバイスに移動
        pred = model(image, question)  # 予測を取得
        pred = pred.argmax(1).cpu().item()  # 最も確からしい予測を取得
        submission.append(pred)  # 予測をリストに追加

    submission = [train_dataset.idx2answer[id] for id in submission]  # インデックスを回答に変換
    submission = np.array(submission)  # 配列に変換
    torch.save(model.state_dict(), "model.pth")  # モデルの重みを保存
    np.save("submission.npy", submission)  # 予測結果を保存

if __name__ == "__main__":
    main()  # メイン関数を実行