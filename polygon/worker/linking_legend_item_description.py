import os
import json
import geopandas as gpd
import numpy as np
import re
from scipy import spatial

from numpy import genfromtxt


def working_geological_time(dir_to_json, map_name, dir_to_gpt4_intermediate, dir_to_info, candidate_info, linking_ids):
    chronology_age = ['Meghalayan', 'Northgrippian', 'Greenlandian', 'Late Pleistocene', 'Chibanian', 'Calabrian', 'Gelasian', 
                    'Piacenzian', 'Zanclean', 'Messinian', 'Tortonian', 'Serravallian', 'Langhian', 'Burdigalian', 'Aquitanian', 
                    'Chattian', 'Rupelian', 'Priabonian', 'Bartonian', 'Lutetian', 'Ypresian', 'Thanetian', 'Selandian', 'Danian', 
                    'Maastrichtian', 'Campanian', 'Santonian', 'Coniacian', 'Turonian', 'Cenomanian', 'Albian', 'Aptian', 'Barremian', 'Hauterivian', 'Valanginian', 'Berriasian', 
                    'Tithonian', 'Kimmeridgian', 'Oxfordian', 'Callovian', 'Bathonian', 'Bajocian', 'Aalenian', 'Toarcian', 'Pliensbachian', 'Sinemurian', 'Hettangian', 
                    'Rhaetian', 'Norian', 'Carnian', 'Ladinian', 'Anisian', 'Olenekian', 'Induan', 
                    'Changhsingian', 'Wuchiapingian', 'Capitanian', 'Wordian', 'Roadian', 'Kungurian', 'Artinskian', 'Sakmarian', 'Asselian', 
                    'Gzhelian', 'Kasimovian', 'Moscovian', 'Bashkirian', 'Serpukhovian', 'Viséan', 'Tournaisian', 
                    'Famennian', 'Frasnian', 'Givetian', 'Eifelian', 'Emsian', 'Pragian', 'Lochkovian', 
                    'Pridoli', 'Ludfordian', 'Gorstian', 'Homerian', 'Sheinwoodian', 'Telychian', 'Aeronian', 'Rhuddanian', 
                    'Hirnantian', 'Katian', 'Sandbian', 'Darriwilian', 'Dapingian', 'Floian', 'Tremadocian', 
                    'Stage 10', 'Jiangshanian', 'Paibian', 'Guzhangian', 'Drumian', 'Wuliuan', 'Stage 4', 'Stage 3', 'Stage 2', 'Fortunian', 
                    'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean'
                    ]
    chronology_epoch = ['Holocene', 'Holocene', 'Holocene', 'late Pleistocene', 'middle Pleistocene', 'middle Pleistocene', 'early Pleistocene', # added layer
                    'Pliocene', 'Pliocene', 'Miocene', 'Miocene', 'Miocene', 'Miocene', 'Miocene', 'Miocene', 
                    'Oligocene', 'Oligocene', 'Eocene', 'Eocene', 'Eocene', 'Eocene', 'Paleocene', 'Paleocene', 'Paleocene', 
                    'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 
                    'Late Jurassic', 'Late Jurassic', 'Late Jurassic', 'Middle Jurassic', 'Middle Jurassic', 'Middle Jurassic', 'Middle Jurassic', 'Early Jurassic', 'Early Jurassic', 'Early Jurassic', 'Early Jurassic', 
                    'Late Triassic', 'Late Triassic', 'Late Triassic', 'Middle Triassic', 'Middle Triassic', 'Early Triassic', 'Early Triassic', 
                    'Lopingian', 'Lopingian', 'Guadalupian', 'Guadalupian', 'Guadalupian', 'Cisuralian', 'Cisuralian', 'Cisuralian', 'Cisuralian', 
                    'Pennsylvanian', 'Pennsylvanian', 'Pennsylvanian', 'Pennsylvanian', 'Mississippian', 'Mississippian', 'Mississippian', 
                    'Late Devonian', 'Late Devonian', 'Middle Devonian', 'Middle Devonian', 'Early Devonian', 'Early Devonian', 'Early Devonian', 
                    'Pridoli', 'Ludlow', 'Ludlow', 'Wenlock', 'Wenlock', 'Llandovery', 'Llandovery', 'Llandovery', 
                    'Late Ordovician', 'Late Ordovician', 'Late Ordovician', 'Middle Ordovician', 'Middle Ordovician', 'Early Ordovician', 'Early Ordovician', 
                    'Furongian', 'Furongian', 'Furongian', 'Miaolingian', 'Miaolingian', 'Miaolingian', 'Series 2', 'Series 2', 'Terreneuvian', 'Terreneuvian', 
                    'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean'
                    ]
    chronology_period = ['Holocene', 'Holocene', 'Holocene', 'Pleistocene', 'Pleistocene', 'Pleistocene', 'Pleistocene', # from epoch
                    'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 
                    'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 
                    'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 
                    'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 
                    'Triassic', 'Triassic', 'Triassic', 'Triassic', 'Triassic', 'Triassic', 'Triassic', 
                    'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 
                    'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 
                    'Devonian', 'Devonian', 'Devonian', 'Devonian', 'Devonian', 'Devonian', 'Devonian', 
                    'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 
                    'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 
                    'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 
                    'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean'
                    ]
    chronology_era = ['Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', # from period
                    'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', # added layer
                    'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', 'Tertiary', # added layer
                    'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 
                    'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 
                    'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 'Mesozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Neoproterozoic', 'Neoproterozoic', 'Neoproterozoic', 'Mesoproterozoic', 'Mesoproterozoic', 'Mesoproterozoic', 'Paleoproterozoic', 'Paleoproterozoic', 'Paleoproterozoic', 'Paleoproterozoic', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean'
                    ]
    chronology_era_eon_prefix = ['Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', # from era
                    'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', # from era
                    'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', 'Cenozoic', # from era
                    'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', 'Late Mesozoic', # adjusted
                    'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', 'Middle Mesozoic', # adjusted
                    'Early Mesozoic', 'Early Mesozoic', 'Early Mesozoic', 'Early Mesozoic', 'Early Mesozoic', 'Early Mesozoic', 'Early Mesozoic', # adjusted
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 'Paleozoic', 
                    'Late Proterozoic', 'Late Proterozoic', 'Late Proterozoic', 'Middle Proterozoic', 'Middle Proterozoic', 'Middle Proterozoic', 'Early Proterozoic', 'Early Proterozoic', 'Early Proterozoic', 'Early Proterozoic', # adjusted
                    'Archean', 'Archean', 'Archean', 'Archean', 'Hadean'
                    ]
    chronology_eon = ['Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 'Phanerozoic', 
                    'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 'Proterozoic', 
                    'Archean', 'Archean', 'Archean', 'Archean', 'Hadean'
                    ]
    chronology_abbr = ['Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 
                    'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 
                    'Pz', 'Pz', 'Pz', 'Pz', 'Pz', 'Pz', 'Pz', 'Pz', 'Pz', 
                    'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 
                    'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 'J', 
                    'T', 'T', 'T', 'T', 'T', 'T', 'T', 
                    'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 
                    'IP', 'IP', 'IP', 'IP', 'M', 'M', 'M', 
                    'D', 'D', 'D', 'D', 'D', 'D', 'D', 
                    'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 
                    'O', 'O', 'O', 'O', 'O', 'O', 'O', 
                    'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 
                    'Z', 'Z', 'Z', 'Y', 'Y', 'Y', 'X', 'X', 'X', 'X', 
                    'A', 'A', 'A', 'A', 'pA'
                    ]

    chronology_age = np.array(chronology_age)
    chronology_epoch = np.array(chronology_epoch)
    chronology_period = np.array(chronology_period)
    chronology_era = np.array(chronology_era)
    chronology_era_eon_prefix = np.array(chronology_era_eon_prefix)
    chronology_eon = np.array(chronology_eon)

    with open(os.path.join(dir_to_json, map_name+'.json')) as f:
        gj = json.load(f)
        for this_gj in gj['shapes']:
            #print(this_gj)
            names = this_gj['label']

            if '_poly' in names:
                this_index = linking_ids[names]
                this_description = candidate_info[this_index][1]+' : '+candidate_info[this_index][0]
                this_description_original = this_description

                b_epoch = chronology_period.shape[0]
                t_epoch = -1
                
                print('-----------------------------------------')
                print(this_description_original)
                this_description = this_description.replace('Upper', 'Late').replace('Lower', 'Early')
                this_description = this_description.replace('upper', 'late').replace('lower', 'early').replace('latest', 'late').replace('?', '')
                
                pattern = r'Middle (and|or|to) Early (\w+)'
                replacement = r'Middle \2 \1 Early \2'
                this_description = re.sub(pattern, replacement, this_description)
                pattern = r'Late (and|or|to) Middle (\w+)'
                replacement = r'Late \2 \1 Middle \2'
                this_description = re.sub(pattern, replacement, this_description)
                pattern = r'middle (and|or|to) early (\w+)'
                replacement = r'middle \2 \1 early \2'
                this_description = re.sub(pattern, replacement, this_description)
                pattern = r'late (and|or|to) middle (\w+)'
                replacement = r'late \2 \1 middle \2'
                this_description = re.sub(pattern, replacement, this_description)
                print(this_description)

                epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_eon)!=-1)
                print(epoch_check)
                if epoch_check.shape[0] > 0:
                    b_epoch = max(epoch_check)
                    t_epoch = min(epoch_check)
                
                epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_era_eon_prefix)!=-1)
                print(epoch_check)
                if epoch_check.shape[0] > 0:
                    b_epoch_temp = b_epoch
                    if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                        b_epoch = min(b_epoch ,max(epoch_check))
                    else:
                        b_epoch = max(b_epoch, max(epoch_check))
                    if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                        t_epoch = max(t_epoch, min(epoch_check))
                    else:
                        t_epoch = min(t_epoch, min(epoch_check))

                epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_era)!=-1)
                print(epoch_check)
                if epoch_check.shape[0] > 0:
                    b_epoch_temp = b_epoch
                    if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                        b_epoch = min(b_epoch ,max(epoch_check))
                    else:
                        b_epoch = max(b_epoch, max(epoch_check))
                    if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                        t_epoch = max(t_epoch, min(epoch_check))
                    else:
                        t_epoch = min(t_epoch, min(epoch_check))
                
                epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_period)!=-1)
                print(epoch_check)
                if epoch_check.shape[0] > 0:
                    b_epoch_temp = b_epoch
                    if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                        b_epoch = min(b_epoch ,max(epoch_check))
                    else:
                        b_epoch = max(b_epoch, max(epoch_check))
                    if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                        t_epoch = max(t_epoch, min(epoch_check))
                    else:
                        t_epoch = min(t_epoch, min(epoch_check))
                
                epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_epoch)!=-1)
                print(epoch_check)
                if epoch_check.shape[0] > 0:
                    b_epoch_temp = b_epoch
                    if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                        b_epoch = min(b_epoch ,max(epoch_check))
                    else:
                        b_epoch = max(b_epoch, max(epoch_check))
                    if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                        t_epoch = max(t_epoch, min(epoch_check))
                    else:
                        t_epoch = min(t_epoch, min(epoch_check))

                epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_age)!=-1)
                print(epoch_check)
                if epoch_check.shape[0] > 0:
                    b_epoch_temp = b_epoch
                    if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                        b_epoch = min(b_epoch ,max(epoch_check))
                    else:
                        b_epoch = max(b_epoch, max(epoch_check))
                    if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                        t_epoch = max(t_epoch, min(epoch_check))
                    else:
                        t_epoch = min(t_epoch, min(epoch_check))
                

                if t_epoch == -1:
                    this_abbr = candidate_info[this_index][1]
                    if len(this_abbr) > 0:
                        this_abbr_shortened = this_abbr[0:min(2,len(this_abbr))]
                        if 'Pz' not in this_abbr_shortened or 'IP' not in this_abbr_shortened or 'pA' not in this_abbr_shortened:
                            this_abbr_shortened = this_abbr[0:min(1,len(this_abbr))]
                        epoch_check = np.flatnonzero(np.core.defchararray.find(this_abbr[0:min(2,len(this_abbr))], chronology_abbr)!=-1)
                        print(epoch_check)
                        if epoch_check.shape[0] > 0:
                            b_epoch_temp = b_epoch
                            if max(epoch_check) < b_epoch and max(epoch_check) > t_epoch:
                                b_epoch = min(b_epoch ,max(epoch_check))
                            else:
                                b_epoch = max(b_epoch, max(epoch_check))
                            if min(epoch_check) < b_epoch_temp and min(epoch_check) > t_epoch:
                                t_epoch = max(t_epoch, min(epoch_check))
                            else:
                                t_epoch = min(t_epoch, min(epoch_check))
                    if t_epoch == -1:
                        b_epoch = chronology_period.shape[0]
                        t_epoch = chronology_period.shape[0]


                concat_name = str(map_name)+'_'+str(names)
                print(concat_name, t_epoch, b_epoch)
                if os.path.isfile(os.path.join(dir_to_info, map_name+'_auxiliary_info.csv')) == False:
                    with open(os.path.join(dir_to_info, map_name+'_auxiliary_info.csv'),'w') as fd:
                        fd.write('Map_name,Key_name')
                        fd.write(',t_epoch,b_epoch')
                        fd.write(',this_description,extract_abbr,extract_desp')
                        fd.write('\n')
                        fd.close()
                with open(os.path.join(dir_to_info, map_name+'_auxiliary_info.csv'),'a') as fd:
                    fd.write(map_name+','+concat_name)
                    fd.write(','+str(t_epoch)+','+str(b_epoch))
                    fd.write(',"'+str(this_description_original)+'","'+str(candidate_info[this_index][1])+'","'+str(candidate_info[this_index][0])+'"')
                    fd.write('\n')
                    fd.close()
                

                if os.path.isfile(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoded.csv')) == False:
                    with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoded.csv'),'w') as fd:
                        fd.write('Map_name,Key_name')
                        for this_time in range(0, chronology_age.shape[0]):
                            fd.write(','+str(chronology_age[this_time]))
                        fd.write(',Missing')
                        fd.write('\n')
                        fd.close()
                with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoded.csv'),'a') as fd:
                    fd.write(map_name+','+concat_name)
                    for this_time in range(0, chronology_age.shape[0]+1):
                        if this_time >= t_epoch and this_time <= b_epoch:
                            fd.write(',1')
                        else:
                            fd.write(',0')
                    fd.write('\n')
                    fd.close()
    return True


def integrate_linking_csv(map_name, dir_to_info):
    if os.path.isfile(os.path.join(dir_to_info, 'auxiliary_info.csv')) == False:
        chronology_age = ['Meghalayan', 'Northgrippian', 'Greenlandian', 'Late Pleistocene', 'Chibanian', 'Calabrian', 'Gelasian', 
                        'Piacenzian', 'Zanclean', 'Messinian', 'Tortonian', 'Serravallian', 'Langhian', 'Burdigalian', 'Aquitanian', 
                        'Chattian', 'Rupelian', 'Priabonian', 'Bartonian', 'Lutetian', 'Ypresian', 'Thanetian', 'Selandian', 'Danian', 
                        'Maastrichtian', 'Campanian', 'Santonian', 'Coniacian', 'Turonian', 'Cenomanian', 'Albian', 'Aptian', 'Barremian', 'Hauterivian', 'Valanginian', 'Berriasian', 
                        'Tithonian', 'Kimmeridgian', 'Oxfordian', 'Callovian', 'Bathonian', 'Bajocian', 'Aalenian', 'Toarcian', 'Pliensbachian', 'Sinemurian', 'Hettangian', 
                        'Rhaetian', 'Norian', 'Carnian', 'Ladinian', 'Anisian', 'Olenekian', 'Induan', 
                        'Changhsingian', 'Wuchiapingian', 'Capitanian', 'Wordian', 'Roadian', 'Kungurian', 'Artinskian', 'Sakmarian', 'Asselian', 
                        'Gzhelian', 'Kasimovian', 'Moscovian', 'Bashkirian', 'Serpukhovian', 'Viséan', 'Tournaisian', 
                        'Famennian', 'Frasnian', 'Givetian', 'Eifelian', 'Emsian', 'Pragian', 'Lochkovian', 
                        'Pridoli', 'Ludfordian', 'Gorstian', 'Homerian', 'Sheinwoodian', 'Telychian', 'Aeronian', 'Rhuddanian', 
                        'Hirnantian', 'Katian', 'Sandbian', 'Darriwilian', 'Dapingian', 'Floian', 'Tremadocian', 
                        'Stage 10', 'Jiangshanian', 'Paibian', 'Guzhangian', 'Drumian', 'Wuliuan', 'Stage 4', 'Stage 3', 'Stage 2', 'Fortunian', 
                        'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                        'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean'
                        ]
        chronology_age = np.array(chronology_age)

        with open(os.path.join(dir_to_info, 'auxiliary_info.csv'),'w') as fd:
            fd.write('Map_name,Key_name')
            for this_time in range(0, chronology_age.shape[0]):
                fd.write(','+str(chronology_age[this_time]))
            fd.write(',Missing')
            fd.write('\n')
            fd.close()
    
    with open(os.path.join(dir_to_info, 'auxiliary_info.csv'),'a') as fd:
        with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoded.csv'),'r') as fdr:
            next(fdr)
            fd.writelines(fdr)
            fdr.close()
        fd.close()



def linking_legend_item_description(map_name, dir_to_json, dir_to_gpt4_intermediate, dir_to_info):
    # Setup referencing text-descriptions
    linking_ids = {}
    candidate_list = []
    candidate_info = []


    with open(os.path.join(dir_to_gpt4_intermediate, map_name+'_polygon.json')) as f:
        gj = json.load(f)
        #print(gj)

        for this_key, this_row in gj.items():
            if ',' in this_key:
                xy_list = this_key[1:-1].split(',')
                center_x = int((float(xy_list[0]) + float(xy_list[2])) / 2.0)
                center_y = int((float(xy_list[1]) + float(xy_list[3])) / 2.0)

                #print(this_key, center_x, center_y, this_row['description'], this_row['symbol name'])
                candidate_list.append([center_x, center_y])
                candidate_info.append([this_row['description'], this_row['symbol name']])
    candidate_list = np.array(candidate_list)
    candidate_info = np.array(candidate_info)


    link_description = True
    if candidate_list.shape[0] > 0:
        with open(os.path.join(dir_to_json, map_name+'.json')) as f:
            gj = json.load(f)
            for this_gj in gj['shapes']:
                #print(this_gj)
                names = this_gj['label']
                features = this_gj['points']

                if '_poly' in names:
                    center_x = int((float(features[0][0]) + float(features[1][0])) / 2.0)
                    center_y = int((float(features[0][1]) + float(features[1][1])) / 2.0)
                    #print(names, center_x, center_y)
                    this_pt = [center_x, center_y]

                    distance,index = spatial.KDTree(candidate_list).query(this_pt)
                    #print(distance, index)

                    #print(names, center_x, center_y, candidate_list[index], distance)
                    if distance < 6.6:
                        #print(candidate_info[index])
                        linking_ids[names] = index
                    else:
                        linking_ids[names] = index
                    pass
    else:
        link_description = False

    print(linking_ids)

    #polygon_type_db = gpd.read_file(path_to_legend_solution, driver='GeoJSON')
    working_geological_time(dir_to_json, map_name, dir_to_gpt4_intermediate, dir_to_info, candidate_info, linking_ids)

    return True, map_name


from transformers import BertTokenizer, BertModel
import torch
import csv

def bert_processing(map_name, dir_to_info):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    #item_description = genfromtxt(os.path.join(dir_to_info, map_name+'_auxiliary_info.csv'), delimiter=',', quotechar='"')
    with open(os.path.join(dir_to_info, map_name+'_auxiliary_info.csv'), newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        item_description = list(reader)
    '''
    item_description = np.array(item_description)

    for row in range(1, item_description.shape[0]):
        this_map_name = item_description[row, 0]
        this_concat_name = item_description[row, 1]
        this_extracted_description = item_description[row, 6]
    '''
    row_counter = 0
    for this_row in item_description:
        row_counter += 1
        if row_counter == 1:
            continue

        item_description = np.array(this_row)
        this_map_name = item_description[0]
        this_concat_name = item_description[1]
        this_extracted_description = item_description[-1]

        input_text = this_extracted_description

        # Encode text
        encoded_input = tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True)

        # Forward pass, get hidden states
        with torch.no_grad():
            outputs = model(encoded_input)
        bert_outputs = outputs.pooler_output
        bert_outputs = bert_outputs.numpy()[0]

        #print(this_extracted_description)
        #print(bert_outputs)
        #print(bert_outputs.shape)

        encoded_input = encoded_input.numpy()[0]
        if os.path.isfile(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoding.csv')) == False:
            with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoding.csv'),'w') as fd:
                fd.write('Map_name,Key_name')
                for this_time in range(0, encoded_input.shape[0]):
                    fd.write(','+'bert_'+str(this_time))
                fd.write('\n')
                fd.close()
        with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoding.csv'),'a') as fd:
            fd.write(map_name+','+this_concat_name)
            for this_time in range(0, encoded_input.shape[0]):
                fd.write(','+str(encoded_input[this_time]))
            fd.write('\n')
            fd.close()

        if os.path.isfile(os.path.join(dir_to_info, map_name+'_auxiliary_info_embedding.csv')) == False:
            with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_embedding.csv'),'w') as fd:
                fd.write('Map_name,Key_name')
                for this_time in range(0, bert_outputs.shape[0]):
                    fd.write(','+'bert_'+str(this_time))
                fd.write('\n')
                fd.close()
        with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_embedding.csv'),'a') as fd:
            fd.write(map_name+','+this_concat_name)
            for this_time in range(0, bert_outputs.shape[0]):
                fd.write(','+str(bert_outputs[this_time]))
            fd.write('\n')
            fd.close()

    if os.path.isfile(os.path.join(dir_to_info, 'encoding_info.csv')) == False:
        with open(os.path.join(dir_to_info, 'encoding_info.csv'),'w') as fd:
            fd.write('Map_name,Key_name')
            for this_time in range(0, encoded_input.shape[0]):
                fd.write(','+'bert_'+str(this_time))
            fd.write('\n')
            fd.close()
    with open(os.path.join(dir_to_info, 'encoding_info.csv'),'a') as fd:
        with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_encoding.csv'),'r') as fdr:
            next(fdr)
            fd.writelines(fdr)
            fdr.close()
        fd.close()

    if os.path.isfile(os.path.join(dir_to_info, 'embedding_info.csv')) == False:
        with open(os.path.join(dir_to_info, 'embedding_info.csv'),'w') as fd:
            fd.write('Map_name,Key_name')
            for this_time in range(0, bert_outputs.shape[0]):
                fd.write(','+'bert_'+str(this_time))
            fd.write('\n')
            fd.close()
    with open(os.path.join(dir_to_info, 'embedding_info.csv'),'a') as fd:
        with open(os.path.join(dir_to_info, map_name+'_auxiliary_info_embedding.csv'),'r') as fdr:
            next(fdr)
            fd.writelines(fdr)
            fdr.close()
        fd.close()


