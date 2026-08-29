TA
#1. Announcement
原先上傳的dataset中audio都是30s片段是錯誤的，目前已更正成全曲audio，並同時更正slide上的資訊.
請各位下載dataset時由雲端連結 https://drive.google.com/drive/folders/1WH8Byx1URaUs5M4FlCwM4VmvZwEm52wJ 下載，請勿透過 Artist20 的官網進行。
我們有事先切好 train / validation / test set，因此如果直接透過 Artist20 的官網進行下載的話會視為作弊。

#2. 
Question: test set 好像有幾首歌是 instrumental
Response: 正常的，目前就挑掉的為 074, 119, 169, 206 這幾首歌的結果將不會影響到成績

#3
Question. 
Task 1 提到的 Feature Extraction 方法是否只能用 Librosa 和 Torchaudio 等套件的函數來處理?
是否可以用 pretrained model 來做 Feature Extraction?

Response.
這項 task 設計是要讓同學去使用一些現有的 rule based features 在解釋性較高的 ML model 去做練習並且討論。
直接使用 pre-trained model 來 extract features 的話可能會讓後面分析變得較複雜
（例如算importance的時候就算知道誰貢獻最大但你不會知道該feature代表甚麼意義）
pre-trained model 可以做為額外的baseline比較，或是在分析時提供較完整的內容來彌補上述說的情況。
（只用 pre-trained model extract features 來完成task1意義不大 因為 task1 的 acc 根本不算分 TA只看分析過程是否合理以及內容是否豐富給分）

#4
Question. Task 2 題目敘述提到 
" Train a deep learning model from scratch"，
想確認這是應用在 classifier 方面的限制還是 encoder 以及 classifier 均不開放使用 pretrained model ? 
另外，若在 pretrained model 上用題目給的資料來做 fine-tunung 是否可以?

Response. 
task2 需要自己重新訓一個 encoder + classifier 喔。
Hw1中有提到使用 pre-trained model (不論encoder還是classifier) 基本上只能作為 baseline。