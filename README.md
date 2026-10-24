# Palco Aberto · karaokê com IA

Karaokê que roda no navegador do celular, com nota em tempo real e uma plateia que reage enquanto você canta.

## Dois modos

**Modo YouTube**: cole o link de um vídeo de karaokê, ou busque pelo nome se tiver uma chave da API do YouTube. O vídeo toca pelo player oficial e o microfone avalia cada frase que você canta:
- **afinação**: quanto as notas sustentadas ficam perto do semitom certo, medida em centésimos (¢)
- **firmeza**: quanto a voz fica estável nas notas longas
- **presença**: quanto da música você realmente cantou

**Modo completo**: para músicas em MP3. O notebook `colab/karaoke_colab.ipynb` prepara cada música:

| Etapa | Ferramenta | Resultado |
|---|---|---|
| Separar a voz | Demucs (htdemucs) | `instrumental.mp3` e `vocals.mp3` |
| Letra sincronizada | LRCLIB | tempo de cada linha |
| Tempo de cada palavra | Whisper (stable-ts) | destaque palavra por palavra |
| Melodia de referência | CREPE | notas usadas na pontuação |

No app, a sua voz é analisada ao vivo com o algoritmo **YIN** e comparada nota a nota com a melodia original. A comparação ignora a oitava, então homem e mulher cantam qualquer música.

## Reações da plateia
Julgamento de cada nota (Perfeito, Ótimo, Bom, Errou), combo, aplauso quando você acerta a linha inteira, assobio a cada 10 de combo, comentários da plateia na tela, celulares acesos quando a galera está no auge e um “uuuh” de leve quando desafina. No fim aparecem a nota de 0 a 100, as estatísticas, um gráfico de acerto por linha e o ranking local.

## Como rodar
O microfone só funciona com HTTPS, então publique no **GitHub Pages**: Settings → Pages → Branch `main` → `/root`.

Não suba arquivos de música para o repositório. As músicas importadas ficam salvas só no aparelho (IndexedDB).

## Tecnologias
HTML, CSS e JavaScript puros · Web Audio API · IndexedDB · YouTube IFrame API · Python (PyTorch, Demucs, Whisper, torchcrepe) no Google Colab
