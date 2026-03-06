"""Train POS, Chunk, and Morphological Tagger using BiLSTM model and data generator using Keras framework and FastText embeddings."""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, concatenate, Activation, Bidirectional, LSTM, Input, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser
# from gensim.models import KeyedVectors
from pickle import load
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from re import findall
from re import S
import fasttext


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def createVectors(lines, wordEmbeddings, char2Index, lcat2Index, gender2Index, number2Index, person2Index, case2Index, vibh2Index, pos2Index, chunk2Index):
    """"Create vectors for words, characters, and tags from the input lines."""
    allSentenceVectors, sentenceVectors = [], []
    allLcatVectors, allGenderVectors, allNumberVectors, allPersonVectors, allCaseVectors, allVibhVectors, allPOSVectors, allChunkVectors = [], [], [], [], [], [], [], []
    lcatsForSent, gendersForSent, numbersForSent, personsForSent, casesForSent, vibhsForSent, posForSent, chunkForSent = [], [], [], [], [], [], [], []
    charSequencesForWords, charSequencesForSentences = [], []
    maxSentenceLength, maxWordLength = 165, 18
    while True:
        for line in lines:
            if line.strip():
                word, lcat, gender, number, person, case, vibh, pos, chunk = line.strip().split('\t')
                charSequenceForWord = [char2Index.get(char, 1) for char in word]
                charSequencesForWords.append(charSequenceForWord)
                charSequenceForWord = []
                sentenceVectors.append(
                    wordEmbeddings.get_word_vector(word).tolist())
                lcatsForSent.append(lcat2Index[lcat] if lcat in lcat2Index else lcat2Index[list(lcat2Index.keys())[0]])
                gendersForSent.append(gender2Index[gender] if gender in gender2Index else gender2Index[list(gender2Index.keys())[0]])
                numbersForSent.append(number2Index[number] if number in number2Index else number2Index[list(number2Index.keys())[0]])
                personsForSent.append(person2Index[person] if person in person2Index else person2Index[list(person2Index.keys())[0]])
                casesForSent.append(case2Index[case] if case in case2Index else case2Index[list(case2Index.keys())[0]])
                vibhsForSent.append(vibh2Index[vibh] if vibh in vibh2Index else vibh2Index[list(vibh2Index.keys())[0]])
                posForSent.append(pos2Index[pos] if pos in pos2Index else pos2Index[list(pos2Index.keys())[0]])
                chunkForSent.append(chunk2Index[chunk] if chunk in chunk2Index else chunk2Index[list(chunk2Index.keys())[0]])
            else:
                if sentenceVectors:
                    allSentenceVectors.append(sentenceVectors)
                    allPOSVectors.append(posForSent)
                    allChunkVectors.append(chunkForSent)
                    sentenceVectors = []
                    allLcatVectors.append(lcatsForSent)
                    allGenderVectors.append(gendersForSent)
                    allNumberVectors.append(numbersForSent)
                    allPersonVectors.append(personsForSent)
                    allCaseVectors.append(casesForSent)
                    allVibhVectors.append(vibhsForSent)
                    lcatsForSent, gendersForSent, numbersForSent, personsForSent, casesForSent, vibhsForSent = [], [], [], [], [], []
                    charSequencesForSentences.append(charSequencesForWords)
                    charSequencesForWords = []
                    charSequenceForWord = []
                    wholeCharSeqForData = []
                    finalSentenceVectors = pad_sequences(
                        allSentenceVectors, padding='post', maxlen=maxSentenceLength)
                    finalLcatSequences = pad_sequences(
                        allLcatVectors, padding='post', maxlen=maxSentenceLength)
                    finalLcatSequences = finalLcatSequences.reshape((-1, maxSentenceLength, 1))
                    finalPOSSequences = pad_sequences(
                        allPOSVectors, padding='post', maxlen=maxSentenceLength)
                    finalPOSSequences = finalPOSSequences.reshape((-1, maxSentenceLength, 1))
                    finalChunkSequences = pad_sequences(
                        allChunkVectors, padding='post', maxlen=maxSentenceLength)
                    finalChunkSequences = finalChunkSequences.reshape((-1, maxSentenceLength, 1))
                    finalGenderSequences = pad_sequences(
                        allGenderVectors, padding='post', maxlen=maxSentenceLength)
                    finalGenderSequences = finalGenderSequences.reshape((-1, maxSentenceLength, 1))
                    finalNumberSequences = pad_sequences(
                        allNumberVectors, padding='post', maxlen=maxSentenceLength)
                    finalNumberSequences = finalNumberSequences.reshape((-1, maxSentenceLength, 1))
                    finalPersonSequences = pad_sequences(
                        allPersonVectors, padding='post', maxlen=maxSentenceLength)
                    finalPersonSequences = finalPersonSequences.reshape((-1, maxSentenceLength, 1))
                    finalCaseSequences = pad_sequences(
                        allCaseVectors, padding='post', maxlen=maxSentenceLength)
                    finalCaseSequences = finalCaseSequences.reshape((-1, maxSentenceLength, 1))
                    finalVibhSequences = pad_sequences(
                        allVibhVectors, padding='post', maxlen=maxSentenceLength)
                    finalVibhSequences = finalVibhSequences.reshape((-1, maxSentenceLength, 1))
                    allSentenceVectors, sentenceVectors = [], []
                    charSequencesForWords = []
                    allLcatVectors, allGenderVectors, allNumberVectors, allPersonVectors, allCaseVectors, allVibhVectors, allPOSVectors, allChunkVectors = [], [], [], [], [], [], [], []
                    for seq in charSequencesForSentences:
                        wholeCharSeqForData.append(pad_sequences(
                            seq, maxlen=maxWordLength, padding='post'))
                    charSequencesForSentences = []
                    finalCharSequences = pad_sequences(
                        wholeCharSeqForData, padding='post', maxlen=maxSentenceLength)
                    wholeCharSeqForData = []
                    allSentenceVectors = []
                    yield [finalCharSequences, finalSentenceVectors], [finalLcatSequences, finalGenderSequences, finalNumberSequences, finalPersonSequences, finalCaseSequences, finalVibhSequences, finalPOSSequences, finalChunkSequences]


def loadObjectToPickleFile(pickleFilePath):
    """Load an object from a pickle file."""
    with open(pickleFilePath, 'rb') as loadFile:
        return load(loadFile)


def createReverseIndex(dictItems):
    """Create a reverse index from a dictionary."""
    return {val: key for key, val in dictItems.items()}


def trainModelUsingBiLSTM(maxWordLen, maxSentLen, totalChars, trainGen, totalLcats, totalGenders, totalNumbers, totalPersons, totalCases, totalVibhs, totalPOS, totalChunks, weightFile, valGen, steps, valSteps, epochs=1):
    """Train a BiLSTM model for POS tagging, chunking, and morphological analysis."""
    embeddingLayer = Embedding(
        totalChars + 2, 50, input_length=maxWordLen, trainable=True)
    charInput = Input(shape=(maxWordLen,))
    charEmbedding = embeddingLayer(charInput)
    charOutput = Bidirectional(LSTM
                               (50, return_sequences=False, dropout=0.3), merge_mode='sum')(charEmbedding)
    charModel = Model(charInput, charOutput)
    charSeq = Input(shape=(maxSentLen, maxWordLen))
    charTD = TimeDistributed(charModel)(charSeq)
    wordSeq = Input(shape=(maxSentLen, 300))
    merge = concatenate([wordSeq, charTD], axis=-1)
    wordSeqLSTM = Bidirectional(LSTM(250, input_shape=(
        maxSentLen, 250), return_sequences=True, dropout=0.3), merge_mode='sum')(merge)
    wordSeqTDForLcat = TimeDistributed(Dense(totalLcats))(wordSeqLSTM)
    activationForLcat = Activation('softmax', name='activationForLcat')(wordSeqTDForLcat)
    wordSeqTDForGender = TimeDistributed(Dense(totalGenders))(wordSeqLSTM)
    activationForGender = Activation('softmax', name='activationForGender')(wordSeqTDForGender)
    wordSeqTDForNumber = TimeDistributed(Dense(totalNumbers))(wordSeqLSTM)
    activationForNumber = Activation('softmax', name='activationForNumber')(wordSeqTDForNumber)
    wordSeqTDForPerson = TimeDistributed(Dense(totalPersons))(wordSeqLSTM)
    activationForPerson = Activation('softmax', name='activationForPerson')(wordSeqTDForPerson)
    wordSeqTDForCase = TimeDistributed(Dense(totalCases))(wordSeqLSTM)
    activationForCase = Activation('softmax', name='activationForCase')(wordSeqTDForCase)
    wordSeqTDForVibh = TimeDistributed(Dense(totalVibhs))(wordSeqLSTM)
    activationForVibh = Activation('softmax', name='activationForVibh')(wordSeqTDForVibh)
    wordSeqTDForPOS = TimeDistributed(Dense(totalPOS))(wordSeqLSTM)
    activationForPOS = Activation('softmax', name='activationForPOS')(wordSeqTDForPOS)
    wordSeqTDForChunk = TimeDistributed(Dense(totalChunks))(wordSeqLSTM)
    activationForChunk = Activation('softmax', name='activationForChunk')(wordSeqTDForChunk)
    finalModel = Model(
        inputs=[charSeq, wordSeq], outputs=[activationForLcat, activationForGender, activationForNumber, activationForPerson, activationForCase, activationForVibh, activationForPOS, activationForChunk])
    finalModel.compile(optimizer='adam',
                       loss={'activationForLcat': 'sparse_categorical_crossentropy',
                             'activationForGender': 'sparse_categorical_crossentropy', 'activationForNumber': 'sparse_categorical_crossentropy',
                             'activationForPerson': 'sparse_categorical_crossentropy', 'activationForCase': 'sparse_categorical_crossentropy',
                             'activationForVibh': 'sparse_categorical_crossentropy', 'activationForPOS': 'sparse_categorical_crossentropy', 'activationForChunk': 'sparse_categorical_crossentropy'},
                       metrics=['accuracy'])
    print(finalModel.summary())
    weightFile = weightFile + '-{epoch:02d}-{val_loss:.2f}.weights.h5'
    checkpointCallback = ModelCheckpoint(
    filepath=weightFile,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='min'
    )
    finalModel.fit(trainGen, steps_per_epoch=steps, epochs=epochs, validation_data=valGen, validation_steps=valSteps, callbacks=[checkpointCallback])
    return finalModel


def main():
    """Main function to parse arguments and initiate training."""
    parser = ArgumentParser(
        description='Train the pos tagging or chunking model')
    parser.add_argument('--train', dest='tr',
                        help='Enter the input file containing pos/chunk labeled data')
    parser.add_argument('--val', dest='val', help='Enter the validation data')
    parser.add_argument('--embed', dest='embed',
                        help='enter word embeddings file')
    parser.add_argument('--c2i', dest='c2i',
                        help='enter char2Index pickle file')
    parser.add_argument('--wt', dest='wt',
                        help='enter weight file')
    args = parser.parse_args()
    if not args.tr or not args.val or not args.embed or not args.c2i or not args.wt:
        print(
            "Usage is not correct. Please provide all the arguments --train, --val, --embed, --c2i, and --wt")
        exit(1)
    else:
        # Create this index2label files from the training data.
        index2Lcat = loadObjectToPickleFile('index-to-lcat-tb.pkl')
        index2Gender = loadObjectToPickleFile('index-to-gender-tb.pkl')
        index2Number = loadObjectToPickleFile('index-to-number-tb.pkl')
        index2Person = loadObjectToPickleFile('index-to-person-tb.pkl')
        index2Case = loadObjectToPickleFile('index-to-case-tb.pkl')
        index2Vibh = loadObjectToPickleFile('index-to-vibh-tb.pkl')
        char2Index = loadObjectToPickleFile(args.c2i)
        index2POS = loadObjectToPickleFile('index-to-pos-tb.pkl')
        index2Chunk = loadObjectToPickleFile('index-to-chunk-tb.pkl')
        lcat2Index = createReverseIndex(index2Lcat)
        pos2Index = createReverseIndex(index2POS)
        chunk2Index = createReverseIndex(index2Chunk)
        gender2Index = createReverseIndex(index2Gender)
        number2Index = createReverseIndex(index2Number)
        person2Index = createReverseIndex(index2Person)
        case2Index = createReverseIndex(index2Case)
        vibh2Index = createReverseIndex(index2Vibh)
        embeddings = fasttext.load_model(args.embed)
        epochs = 20
        batchSize = 16
        print('--TRAIN GEN--')
        trainLines = read_lines_from_file(args.tr)
        if trainLines[-1] != '\n':
            trainLines = trainLines + ['\n']
        totalSamples = len(findall('\n\n', ''.join(trainLines) + '\n', S))
        print('Total Samples', totalSamples)
        if totalSamples % batchSize == 0:
            steps = totalSamples // batchSize
        else:
            steps = totalSamples // batchSize + 1
        print(steps, 'STEPS')
        trainGen = createVectors(
            trainLines, embeddings, char2Index, lcat2Index, gender2Index, number2Index, person2Index, case2Index, vibh2Index, pos2Index, chunk2Index)
        valLines = read_lines_from_file(args.val)
        if valLines[-1] != '\n':
            valLines = valLines + ['\n']
        valSamples = len(findall('\n\n', ''.join(valLines) + '\n', S))
        print('Total Val Samples', valSamples)
        if valSamples % batchSize == 0:
            valSteps = valSamples // batchSize
        else:
            valSteps = valSamples // batchSize + 1
        print(valSteps, 'VAL STEPS')
        valGen = createVectors(
            valLines, embeddings, char2Index, lcat2Index, gender2Index, number2Index, person2Index, case2Index, vibh2Index, pos2Index, chunk2Index)
        del embeddings
        maxWordLength, maxSentenceLength = 18, 165
        print('MAX word, max Sent', maxWordLength, maxSentenceLength)
        biLSTM = trainModelUsingBiLSTM(maxWordLength, maxSentenceLength, len(char2Index), trainGen, len(lcat2Index), len(gender2Index), len(number2Index), len(person2Index), len(case2Index), len(vibh2Index), len(pos2Index), len(chunk2Index), args.wt, valGen, steps, valSteps, epochs=epochs)


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    main()
