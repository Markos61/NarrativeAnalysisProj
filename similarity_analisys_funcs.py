##
from natasha import MorphVocab
from natasha import NewsNERTagger
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)
from add_content_func import add_context
from writer_func import writer
from similarity_count import similarity_finding
from similarity_ranging import narratives_ranging
import streamlit as st


def formalize_text(x: list, tokenizer, model):
    """
    Функция, которая ищет нарративные признаки
    :param model: Модель, для получения тональности;
    :param tokenizer: Токенайзер;
    :param x: список строк с текстом;
    """
    triads = []
    doc_count = -1
    for index, single_text in enumerate(x):
        doc_count += 1
        segmenter = Segmenter()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        syntax_parser = NewsSyntaxParser(emb)
        doc = Doc(single_text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        morph_vocab = MorphVocab()
        ner_tagger = NewsNERTagger(emb)
        doc.tag_ner(ner_tagger)

        # Спикер (актор)
        current_speaker = ''

        # Дата
        date = ''

        sent_count = 0  # Счётчик, чтобы отследить первое предложение для поиска спикера
        for sent in doc.sents:
            sent_count += 1
            sent_for_analysis = sent  # Предложение для анализа
            # Идея -- использовать данный цикл для нахождения отправных слов (индексов)

            # Словарь для записи триады нарратива
            triad = {'speakers': [], 'actors': [], 'actions': [], 'objects': [],
                     'modality': [], 'tonality': [], 'sentence': [], 'path': [],
                     'date': [], 'connected sentences': []}

            # Список с исходными элементами поиска триады
            narrative_tokens = []
            for token in sent.tokens:

                # Добавление действия (идеи), начальная точка поиска
                if token.rel == "neg" or token.rel == "root" or (token.pos == 'VERB' and token.rel != 'csubj'):
                    # не добавляем управляемое сказуемое без подлежащего
                    if token.rel != 'xcomp':
                        narrative_tokens.append(token)

                elif token.rel == "parataxis" and token.pos != 'ADV':
                    # Если есть такой элемент, то предложение назывное
                    narrative_tokens.append(token)

                elif token.id == token.head_id and token.pos == 'NOUN':
                    narrative_tokens.append(token)

                else:
                    pass

            # Если в предложении нет сказуемого и parataxis, ищем подлежащее
            if not narrative_tokens:
                for token in sent.tokens:
                    if token.rel == "nsubj" or token.rel == "nsubj:pass":
                        narrative_tokens.append(token)

            # Конец поиска, у нас есть список с исходными токенами
            actors, actions = [[] for _ in range(len(narrative_tokens))], [[] for _ in range(len(narrative_tokens))]
            objects = [[] for _ in range(len(narrative_tokens))]
            modality = [[] for _ in range(len(narrative_tokens))]
            tonality = [[] for _ in range(len(narrative_tokens))]
            date_list = [date for _ in range(len(narrative_tokens))]

            speakers = [current_speaker for _ in range(len(narrative_tokens))]
            # Собираем триаду перебором начальных элементов
            for narrative_num, n_token in enumerate(narrative_tokens):
                # Выбор пути
                if n_token.rel == "neg" or n_token.rel == "root" or n_token.pos == 'VERB':
                    # ПУТЬ №1 ГЛАГОЛА-СКАЗУЕМОГО
                    if (n_token.rel == 'root' and n_token.pos == 'NOUN') or (
                            n_token.rel == 'root' and n_token.pos == 'PROPN'):
                        # Если root'ом оказалось существительное
                        actions[narrative_num].append(n_token.text)
                        noun_id = n_token.id
                        # Ищем зависимые элементы
                        for p_pod in sent.tokens:
                            # Ищем соответствующее однородное (Государство - источник)
                            if p_pod.rel == "conj" or p_pod.rel == 'nsubj':
                                if p_pod.head_id == noun_id:
                                    actors[narrative_num].append(p_pod.text)

                    else:
                        # Если не root или root'ом оказалось не существительное
                        actions[narrative_num].append(n_token.text)
                        glag_id = n_token.id
                        # Ищем зависимые элементы
                        for p_pod in sent.tokens:
                            # Ищем соответствующие подлежащее
                            if p_pod.rel == "nsubj" or p_pod.rel == "nsubj:pass":
                                if p_pod.head_id == glag_id:
                                    actors[narrative_num].append(p_pod.text)

                            # Ищем Управляемое сказуемое (придёт РАБОТАТЬ)
                            if p_pod.rel == "xcomp" or p_pod.rel == 'csubj':
                                if p_pod.head_id == glag_id:
                                    actions[narrative_num].append(p_pod.text)

                            # Поиск частицы с негативной окраской
                            elif p_pod.rel == "advmod":
                                if p_pod.head_id == glag_id:
                                    try:
                                        if p_pod.feats['Polarity'] == "Neg":
                                            actions[narrative_num].insert(0, p_pod.text)
                                            # triad['actions'][-2:] = [" ".join(triad['actions'][-2:])]
                                    except KeyError:
                                        pass

                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                elif n_token.rel == "parataxis" and n_token.pos != 'ADV':
                    # ПУТЬ №2
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                elif n_token.rel == "nsubj" or n_token.rel == "nsubj:pass":
                    # ПУТЬ №3 БЕЗ ГЛАГОЛА
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                elif n_token.id == n_token.head_id and n_token.pos == 'NOUN':
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                else:
                    pass

            triad['speakers'] = speakers
            triad['actors'] = actors
            triad['actions'] = actions
            triad['objects'] = objects
            triad['modality'] = modality
            triad['tonality'] = tonality
            triad['sentence'] = sent.text
            # triad['path'] = doc_paths[index]
            # triad['date'] = date_list

            start = max(0, sent_count - 3)
            end = min(len(doc.sents) - 1, sent_count + 3)
            con_sents = ''
            for idx, con_sent in enumerate(doc.sents[start:end]):
                con_sents = con_sents + " " + con_sent.text
            triad['connected sentences'] = con_sents

            triads.append(triad)

    return triads


@st.cache_data
def similarity_start(x: list, narratives: list):
    if not narratives:
        return 'Введите текст для того, чтобы начать анализ'
    res = formalize_text(x, '', '')
    res1 = writer(res, 'file')
    res2 = similarity_finding(res1, narratives)
    res3 = narratives_ranging(res2)
    return res3
