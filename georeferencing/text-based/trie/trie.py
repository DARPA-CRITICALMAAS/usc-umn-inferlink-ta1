from typing import Tuple
import string 
import json
import pdb
import argparse
import pandas as pd
import geopandas as gpd
import sys
import re
import os
# Reference:
# https://towardsdatascience.com/implementing-a-trie-data-structure-in-python-in-less-than-100-lines-of-code-a877ea23c1a1
# https://stackoverflow.com/questions/23595801/how-to-serialize-a-tree-class-object-structure-into-json-file-format

class TrieNode(dict):
    '''
    # save a Trie
    json_str = json.dumps(root, indent=2)
    json_str = json.dump(root, f, indent=2)
    
    # load a Trie from json str
    pyobj = TrieNode.from_dict(json.loads(json_str))  # reconstitute
    
    '''
    
    def __init__(self, word, children=None, phrase_finished = False, counter = 1):
        super().__init__()
        self.__dict__ = self
        self.word = word
        self.children = list(children) if children is not None else []
        # Is it the last character of the word.`
        self.phrase_finished = phrase_finished
        # How many times this character appeared in the addition process
        self.counter = counter

    @staticmethod
    def from_dict(dict_):
        """ Recursively (re)construct TrieNode-based tree from dictionary. """
        node = TrieNode(dict_['word'], dict_['children'], dict_['phrase_finished'],dict_['counter'])
        node.children = list(map(TrieNode.from_dict, node.children))
        return node 


    def __repr__(self):
        def recur(node, indent):
            return "".join(indent + child.word + "" 
                                + recur(child, indent + "  ") 
                for child in node.children )

        return recur(self, "\n")
    

        # def recur(node, indent):
        #     return "".join(indent + key + "") 
        #                           + recur(child, indent + "  ") 
        #         for key, child in node.children.items())

        # return recur(self.root, "\n")

    

def trie_add_node(root, phrase: list):
    """
    Adding a phrase in the trie structure
    """
    node = root
    for word in phrase:
        found_in_child = False
        # Search for the word in the children of the present `node`
        for child in node.children:
            if child.word == word:
                # We found it, increase the counter by 1 to keep track that another
                # phrase has it as well
                child.counter += 1
                # And point the node to the child that contains this word
                node = child
                found_in_child = True
                break
        # We did not find it so add a new chlid
        if not found_in_child:
            new_node = TrieNode(word)
            node.children.append(new_node)
            # And then point node to the new child
            node = new_node
    # Everything finished. Mark it as the end of a phrase.
    node.phrase_finished = True

def print_string(idx, stri):
    print(idx, stri)

def write_to_dict(out_dict, idx, stri):
    stri = ' '.join(stri)
    if stri in out_dict:
        out_dict[stri].append(idx)
    else:
        out_dict[stri] = [idx]

    return out_dict

def trie_find_prefix(root, prefix: list) -> Tuple[bool, int]:
    """
    Check and return 
      1. If the prefix exsists in any of the words we added so far
      2. If yes then how may words actually have the prefix
    """
    node = root
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie
    if not root.children:
        return False, False
    for word in prefix:
        word_not_found = True
        # Search through all the children of the present `node`
        for child in node.children:
            if child.word == word:
                # We found the word existing in the child.
                word_not_found = False
                # Assign node as the child containing the word and break
                node = child
                break
        # Return False anyway when we did not find a word.
        if word_not_found:
            return False, False
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    return True, node.phrase_finished

def trie_find_entities(trie_root, input_str_list, if_print = True):
    start_idx = 0
    out_dict = dict()
    for idx in range(len(input_str_list)):
        if idx < start_idx:
            continue # if current word is the >2nd word in a location phrase

        word = input_str_list[idx]

        first_found, first_complete = trie_find_prefix(trie_root, [word])

        is_complete_list = [first_complete]
        if not trie_find_prefix(trie_root, [word])[0]:
            # no entity name starts with this word
            # continue to check if exists entity with name starts with next word
            continue 
        else:
            if idx +1 == len(input_str_list):
                # single-word location name at the end of sentence.
                if if_print:
                    print_string(idx, [word])
                write_to_dict(out_dict, idx, [word])
            else:
            
                is_reach_end = True 

                for end_idx in range(idx+1, len(input_str_list)): # end_idx should be one position pass the sentence length
                    #print(input_str_list[idx:end_idx])
                    #pdb.set_trace()
                    is_found, is_complete = trie_find_prefix(trie_root, input_str_list[idx:end_idx+1])
                    is_complete_list.append(is_complete)
                    if is_found == False: # stop looking at the next word if pattern does not match anymore
                        is_reach_end = True if end_idx == len(input_str_list) else False
                        break


                # loop ends because end_idx reach the length of the input string
                if is_reach_end:
                    if is_complete_list[-1]:
                        if if_print:
                            print_string(idx, input_str_list[idx:end_idx+1])
                        write_to_dict(out_dict, idx, input_str_list[idx:end_idx+1])
                        start_idx = end_idx+1 # move cursor to the next new word
                    else:
                        if True in is_complete_list: # whether match exist in substring
                            sub_end_idx = max(loc for loc, val in enumerate(is_complete_list) if val == True)
                            if if_print:
                                print_string(idx, input_str_list[idx:idx+sub_end_idx+1])
                            write_to_dict(out_dict, idx, input_str_list[idx:idx+sub_end_idx+1])
                            start_idx = idx + sub_end_idx +1

                else: # loop ends because is_found == False 

                    # found longest match, check if the match is a complete place name
                    if is_complete_list[-2] == True:  
                        if if_print:
                            print_string(idx, input_str_list[idx:end_idx])
                        write_to_dict(out_dict, idx, input_str_list[idx:end_idx])
                        start_idx = end_idx # move cursor to the next new word
                    else:
                        if True in is_complete_list: # whether match exist in substring
                            sub_end_idx = max(loc for loc, val in enumerate(is_complete_list) if val == True)
                            if if_print:
                                print_string(idx, input_str_list[idx:idx+sub_end_idx+1])
                            write_to_dict(out_dict, idx, input_str_list[idx:idx+sub_end_idx+1])
                            start_idx = idx + sub_end_idx + 1
                  
    return out_dict


def clean_text(s):
    s = s.lower()
    # Define a regex pattern to delete all punctuation
    pattern = r'[^\w\s]'
    return re.sub(pattern, ' ', s)

def find_matching_topos(geomaps, topo_terms):
    '''
    Function will create a trie using the topo_terms list and check the geomap titles to see if there's any corresponding terms
    
    :param geomaps: geomaps should be dict of geomap map_names and corresponding title
    :param topos: topo_terms should be list of strings that need to be added to trie tree
    
    :return: dict of (geomap, [topo_terms])
    '''
    
    #based on the attribute, load every term in the topo file into the tree
    root = TrieNode('*')
    for placename in topo_terms:
        if placename is None:
            continue
        placename_aslist = placename.split(',')
        for term in placename_aslist:
            termlist = term.split()
            trie_add_node(root, termlist)

    #For every map in the set, save a list of the topo maps corresponding
    geomap_matches_outdict = {}
    for geomap, title in geomaps.items():
        input_str = title
        input_str_list = input_str.split(' ')
        input_str_list = [clean_text(word) for word in input_str_list]
        print(input_str_list)
        out_dict=trie_find_entities(root, input_str_list, if_print=False)
        placenames_in_geomap = list(out_dict.keys())
        geomap_matches_outdict[geomap] = placenames_in_geomap
        
    # json_str = json.dumps(root, indent=2)
    # pyobj = TrieNode.from_dict(json.loads(json_str))
    # print('PYR OBJ: ', pyobj)
        
    return geomap_matches_outdict
    
def run_trie_ocr(geomap_metadata_pt, ocr_root,  topo_geojson_pt, cand_out_pt, trie_analysis_out_pt):
    geomap_df = pd.read_csv(geomap_metadata_pt)
    geomap_names = geomap_df.map_name.tolist()
    
    geomap_dict = {}
    for map_name in geomap_names:
        map_text = ''
        ocr_pt = os.path.join(ocr_root, map_name +'.geojson')
        if not os.path.exists(ocr_pt):
            continue
        with open(ocr_pt, 'r') as file:
            data = json.load(file)
            for item in data['features']:
                properties = item.get('properties', {})
                text = properties.get('text', '')
                if text:
                    map_text += text + " "
            geomap_dict[map_name] = clean_text(map_text)
    
    #Find state matches
    topo_df = gpd.read_file(topo_geojson_pt)
    topo_df['map_scale'] = topo_df['map_scale'].astype(str)
    columns_to_clean = ['primary_state', 'county_list', 'map_name', 'map_scale']
    topo_df.dropna(subset=columns_to_clean, inplace=True)
    topo_df[columns_to_clean] = topo_df[columns_to_clean].applymap(clean_text)
    topo_terms = topo_df['primary_state'].tolist()
    state_matches = find_matching_topos(geomap_dict, topo_terms)
    
    #Find quadrangle matches
    flattened_states = [value for sublist in state_matches.values() for value in sublist]
    filtered_topo = topo_df[topo_df['primary_state'].isin(flattened_states)]
    topo_terms = filtered_topo['map_name'].tolist()
    quad_matches = find_matching_topos(geomap_dict, topo_terms)
    
    #Find county matches
    flattened_quads = [value for sublist in quad_matches.values() for value in sublist]
    filtered_topo = filtered_topo[filtered_topo['map_name'].isin(flattened_quads)]
    topo_terms = filtered_topo['county_list'].tolist()
    
    county_matches = find_matching_topos(geomap_dict, topo_terms)
    trie_output_dict = {}
    for geomap, scale in quad_matches.items(): #need to change when enabling/disabling attributes
        target_state = state_matches[geomap]
        target_county = county_matches[geomap]
        target_quad = quad_matches[geomap]
        # print('county list: ', filtered_topo['county_list'])
        result = filtered_topo[(filtered_topo['primary_state'].isin(target_state)) & 
                               (filtered_topo['county_list'].str.contains('|'.join(target_county))) &
                               (filtered_topo['map_name'].isin(target_quad))]
        # result = filtered_topo[(filtered_topo['primary_state'].isin(target_state)) & 
        #                        (filtered_topo['map_name'].isin(target_quad))]
        topo_names = result.product_url.tolist()
        topo_names = [mapname.split('/')[-1].replace('.pdf', '') for mapname in topo_names]
        trie_output_dict[geomap] = topo_names
        
    df = pd.DataFrame(trie_output_dict.items(), columns=['geomap_name', 'topos'])

    # Save DataFrame to CSV
    df.to_csv(cand_out_pt, index=False)
    
    # Save matches to CSV
    df_matches = pd.DataFrame({'map_title': list(geomap_dict.keys()),
                   'state': list(state_matches.values()),
                   'quadrangle': list(quad_matches.values()),
                   'county': list(county_matches.values())})
    df_matches.to_csv(trie_analysis_out_pt, index=False)
    
def run_trie_GPT():
    geomap_df = pd.read_csv('/home/yaoyi/chen7924/critical-maas/models/text_extraction/outputs/gpt_title_e2.csv')
    geomap_df_1 = geomap_df[['map_name','title']]
    geomap_df_1 = geomap_df_1.head(10) # FOR TESTING PURPOSES ONLY
    geomap_dict_raw = geomap_df_1.set_index('map_name')['title'].to_dict()
    geomap_dict = {key_str:clean_text(val_str) for key_str, val_str in geomap_dict_raw.items()}
    
    #Find state matches
    topo_df = gpd.read_file('/home/yaoyi/chen7924/critical-maas/Data/usgs_topo_geoms.geojson')
    columns_to_clean = ['primary_state', 'county_list', 'map_name']
    topo_df.dropna(subset=columns_to_clean, inplace=True)
    topo_df[columns_to_clean] = topo_df[columns_to_clean].applymap(clean_text)
    topo_terms = topo_df['primary_state'].tolist()
    state_matches = find_matching_topos(geomap_dict, topo_terms)
    
    #Find quadrangle matches
    flattened_states = [value for sublist in state_matches.values() for value in sublist]
    filtered_topo = topo_df[topo_df['primary_state'].isin(flattened_states)]
    topo_terms = filtered_topo['map_name'].tolist()
    quad_matches = find_matching_topos(geomap_dict, topo_terms)
    
    #Find county matches
    flattened_quads = [value for sublist in quad_matches.values() for value in sublist]
    filtered_topo = filtered_topo[filtered_topo['map_name'].isin(flattened_quads)]
    topo_terms = filtered_topo['county_list'].tolist()
    county_matches = find_matching_topos(geomap_dict, topo_terms)
    
    trie_output_dict = {}
    for geomap, _ in county_matches.items(): #need to change when enabling/disabling attributes
        target_state = state_matches[geomap]
        target_county = county_matches[geomap]
        target_quad = quad_matches[geomap]
        result = filtered_topo[(filtered_topo['primary_state'].isin(target_state)) & 
                               (filtered_topo['county_list'].str.contains('|'.join(target_county))) &
                               (filtered_topo['map_name'].isin(target_quad))]
        # result = filtered_topo[(filtered_topo['primary_state'].isin(target_state)) & 
        #                        (filtered_topo['map_name'].isin(target_quad))]
        topo_names = result.product_url.tolist()
        topo_names = [mapname.split('/')[-1].replace('.pdf', '') for mapname in topo_names]
        trie_output_dict[geomap] = topo_names
        
    df = pd.DataFrame(trie_output_dict.items(), columns=['geomap_name', 'topos'])

    # Save DataFrame to CSV
    df.to_csv('outputs/trie_state-quad-county_100testset_gpt2.csv', index=False)
    
    # Save matches to CSV
    df_matches = pd.DataFrame({'map_title': list(geomap_dict.values()),
                   'state': list(state_matches.values()),
                   'quadrangle': list(quad_matches.values()),
                   'county': list(county_matches.values())})
    df_matches.to_csv('outputs/MATCHES_trie_state-quad-county_100testset_gpt2.csv', index=False)

if __name__ == "__main__":
    geomap_metadata_pt = '/home/yaoyi/shared/critical-maas/testset_100map/100map_testset_metadata.csv'
    ocr_root = '/home/yaoyi/shared/critical-maas/testset_100map/geomaps_ocr'
    topo_geojson = '/home/yaoyi/chen7924/critical-maas/Data/usgs_topo_geoms.geojson'
    cand_out_pt = 'outputs/trie_state-quad_100testset_ocr.csv'
    trie_analysis_out_pt = 'outputs/MATCHES_trie_state-quad_100testset_ocr.csv'
    # run_trie_GPT()
    run_trie_ocr(geomap_metadata_pt, ocr_root, topo_geojson, cand_out_pt, trie_analysis_out_pt)