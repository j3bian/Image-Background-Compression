

from registry import *
from problem import *
from interpret import *

import sys
import os
from pygame import *
import numpy
import grammar_v5 as grammar
import text_encode as encode


class StandardProblem(Problem):

    __attributes__ = ['registry', # a Registry that tells us what shape names correspond to what class
                      'object_min', # minimal amount of objects (this does nothing for the moment)
                      'object_max', # maximal amount of objects
                      'palette', # list of RGB tuples
                      'color_names', # names of the colors
                      ('area_elong_update', True), # True if the shapes will be updated with area and elongation
                      'shape_names', # names of the possible shapes
                      'shape_samplers', # list of dictionaries associating each shape attribute to a Sampler
                      'x_res_view', # width of the image for viewing
                      'y_res_view', # height of the image for viewing
                      ('scene_generator', SceneGenerator),
                      'x_value', # class to interpret x coordinates
                      'y_value', # class to interpret y coordinates
                      'area_value', # class to interpret areas
                      'elong_value', # class to interpret elongations
                      'xcmp_value', # class to interpret the difference in x between two objects
                      'ycmp_value', # class to interpret the difference in y between two objects
                      'areacmp_value', # class to interpret the area ratio between two objects
                      'shape_probs', # a prori probability of each shape
                      'rng', # random number generator
                      'language_question',
                      ('language_real_question', {'color': 1, 'shape': 1, 'location_hor': 1, 'location_vert': 1, 'size': 1}),
                      'language_sentence',
                      'language_objects',
                      'language_form',
                      'language_background',
                      'language_negation',
                      ('histogram_ceil', 1),
                      ('text_encodings', ['o', 1, 2, 3]), # a list of: 'o' (onehot), 'i' (int), 1, 2, 3 ... (-grams)
                      ('question_words', []),
                      ('answer_words', []),
                      ('gram_dictionaries', []),
                      ('number_words_max', 0)]


    def __init__(self, **args):
        Problem.__init__(self, **args)
        if self.palette:
            self.registry.classof('__base__').__palette__ = self.palette


    @cacheresult
    def __value_cls__(self):

        p = self

        TVC = lambda x: TableValueClass(x, self.rng)

        
        ##### GENERAL #####
        # p.object_min/max = minimal/maximal amount of shapes in the scene
        # p.shape_types = list of possible shapes that can be generated
        # p.shape_names = names for the shapes

        ntypes = len(p.shape_names)
        sgens = self.__shape_generators__()
        TypeValue = TVC([(prob, sgen, oh(ntypes, i), name) for prob, sgen, i, name in zip(p.shape_probs, sgens, xrange(ntypes), p.shape_names)])
        

        ##### COLOR #####
        # p.palette = list of colors or None if p.color_bands is used
        # p.color_bands = list of color bands: each element is a (min, max) pair; None if p.palette is used
        # p.color_names = names for each color band or palette color that can be generated

        if p.palette:
            ncols = len(p.palette)
            ColorValue = TVC(equiprob(
                [(i, oh(ncols, i), name) for color, i, name in zip(p.palette, xrange(ncols), p.color_names)]))
        else:
            ncols = len(p.color_bands)
            ColorValue = TVC(equiprob(
                [(MultiIntUniformSampler(*(zip(color_min, color_max) + (self.rng and [self.rng] or []))),
                  oh(ncols, i),
                  name) for (color_min, color_max), i, name in zip(p.palette, xrange(ncols), p.color_names)]))

        
        ##### ELONGATION ####
        ElongationValue = TVC(p.elong_value)

        ##### AREA #####
        AreaValue = TVC(p.area_value)

        ##### LOCATION #####
        XValue = TVC(equiprob(p.x_value))
        YValue = TVC(equiprob(p.y_value))

        ##### COMPARISON #####
        AreacmpValue = TVC(equiprob(p.areacmp_value))
        XcmpValue = TVC(equiprob(p.xcmp_value))
        YcmpValue = TVC(equiprob(p.ycmp_value))

        ##### BUILD MAPPINGS #####
        # All local variables with a name ending in Value will have their content associated
        # to the first part of their name in the dict returned from this function.
        # Example: if somewhere in the function there is: ColorValue = xyz, then: d['color'] = xyz
        
        d = {}
        suffix = 'Value'
        for attr in filter(lambda local: local.endswith(suffix), locals()):
            d[attr[:-len(suffix)].lower()] = eval(attr)
        return d


    @cacheresult
    def __shape_generators__(self):
        p = self
        return [ShapeGenerator(p.registry, samplers) for samplers in p.shape_samplers]
    
    @cacheresult
    def __generator__(self):
        
        class FirstIsDifferentSampler(ConstrainedSampler):
            @staticmethod
            def valid(a):
                # valid only if the first element isn't found in the rest
                return a.count(a[0]) == 1

        vc = self.__value_cls__()

        p = self
        nshapes = p.object_max
        
        return p.scene_generator(
            p.registry,
            nshapes,
            # Color generator
            FirstIsDifferentSampler(MultiSampler([vc['color'].generator_exact()] * (nshapes + 1))),
            # Texture generator
            MultiSampler([ConstantSampler(None)] * (nshapes + 1)),
            # Shape generator
            MultiSampler([vc['type'].generator_exact()] * nshapes),
            # Area generator
            p.area_elong_update and MultiSampler([vc['area'].generator_exact()] * nshapes),
            # Elongation generator
            p.area_elong_update and MultiSampler([vc['elongation'].generator_exact()] * nshapes),
            # Position generator
            None,
            p.rng)

    def set_seed(self, seed):
        self.__generator__().set_seed(seed)


    def generate(self):
        return self.__generator__().generate()


    def load(self, desc):
        return self.registry.load(desc)

    def surface(self, scene):
        return scene.surface(self.x_res_view, self.y_res_view)

    def vector(self, scene):
        p = self
        mat = scene.matrix(p.x_res, p.y_res)
        mat = mat.copy()
        ret = mat.resize((p.x_res * p.y_res, ))
        if ret is None:
            raise Exception("numpy.resize did not returned what we expected.")
        else: return ret

    def describe(self, scene):
        vc = self.__value_cls__()
        iscene = self.interpret(scene)
        d = {}
        d['background'] = dict(color = iscene.color.english(),
                               color_value = iscene.color.exact())
        d['veracity'] = 'TRUE'
        nshapes = len(iscene.shapes)
        for i in xrange(nshapes):
            # The interpretation wrapper doesn't handle iterators, so 'for shape in interp.shapes' won't work
            shape = iscene.shapes[i] # it does handle __getitem__, though!
            key = 'object_' + str(i)
            d[key] = dict(shape = shape.type.english(),
                          hrzpos = shape.x.english(),
                          hrzpos_value = shape.x.exact(),
                          vrtpos = shape.y.english(),
                          vrtpos_value = shape.y.exact(),
                          color = shape.color.english(),
                          color_value = shape.color.exact(),
                          size = shape.area().english(),
                          size_value = shape.area().exact())
            for j in xrange(i+1, nshapes):
                shape2 = scene.shapes[i]
                shape1 = scene.shapes[j]
                def comparison(property, value):
                    return vc[property + 'cmp'].from_exact(value)
                hrzpos = comparison('x', shape1.x - shape2.x)
                vrtpos = comparison('y', shape1.y - shape2.y)
                size = comparison('area', shape1.area() / shape2.area())
                key = 'comparison_' + str(i) + "/" + str(j)
                d[key] = dict(hrzpos = hrzpos.english(),
                              hrzpos_value = hrzpos.exact(),
                              vrtpos = vrtpos.english(),
                              vrtpos_value = vrtpos.exact(),
                              size = size.english(),
                              size_value = size.exact())
        return d


    def interpret(self, scene):
        if not scene:
            raise "Can't interpret this."
        elif isinstance(scene, Scene):
            d = {}
            for (attr, valuecls) in self.__value_cls__().items():
                d[attr] = valuecls.from_exact
            return AttributeInterpreter(scene, d)
        else:
            return self.interpret(self.load(scene))

    def text(self, scene):
        if scene.text:
            return [scene.text]
        else:
            texts = grammar.describe(self.describe(scene),
                                     input_file = None,
                                     number_objects = self.object_max,
                                     output_file = None,
                                     language_question = self.language_question,
                                     language_sentence = self.language_sentence,
                                     language_objects = self.language_objects,
                                     language_form = self.language_form,
                                     language_background = self.language_background,
                                     language_negation = self.language_negation)
            if self.language_sentence == 'oneofeach':
                iscene = self.interpret(scene)
                texts_ = ['']*5
                for text in texts:
                    answer = text.split('?')[-1].strip()
                    if answer in [iscene.shapes[i].type.english() for i in xrange(len(iscene.shapes))]:
                        texts_[0] = text
                    elif answer in [iscene.shapes[i].color.english() for i in xrange(len(iscene.shapes))]:
                        texts_[1] = text
                    elif answer in [iscene.shapes[i].area().english() for i in xrange(len(iscene.shapes))]:
                        texts_[2] = text
                    elif answer in [iscene.shapes[i].x.english() for i in xrange(len(iscene.shapes))]:
                        texts_[3] = text
                    elif answer in [iscene.shapes[i].y.english() for i in xrange(len(iscene.shapes))]:
                        texts_[4] = text
                texts = texts_
#                 print texts
                for attr,i in zip(['shape', 'color', 'size', 'location_hor', 'location_vert'], xrange(5)):
                    if not self.language_real_question[attr]:
                        texts[i] = None
                texts = [t for t in texts if t != None]
#                 print texts
#                 print "---"
                
            return texts

    def text_encode(self, scene, expand_onehot = True):
        texts = self.text(scene)
        grams = [x for x in self.text_encodings if isinstance(x, int)]
        grams.sort()
        def do_encode(enc, q):
            if enc == 'o':
                if expand_onehot:
                    return encode.question_onehot_encoding(q)
                else:
                    return encode.question_int_encoding(q)
            elif enc == 'i':
                return encode.question_int_encoding(q)
            elif isinstance(enc, int):
                i = grams.index(enc)
                gram_list = self.gram_dictionaries[i][0]
                gram_index = self.gram_dictionaries[i][1]
                size_gram = len(gram_list)
                encode.set_globals(gramLIST = gram_list,
                                   gramINDEX = gram_index,
                                   SIZE_gram = size_gram,
                                   gram_size = enc)
                return [min(self.histogram_ceil, n) for n in encode.question_gram_encoding(q)]
        l = []
        for text in texts:
            q, a = encode.read_questionAnswer(text)
            l.append([do_encode(enc, q) for enc in self.text_encodings] + [encode.answer_int_encoding(a)])
        return l
