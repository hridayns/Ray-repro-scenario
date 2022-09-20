import argparse
from audioop import add
import os, sys
import re

try:
    import xml.etree.cElementTree as ET
except ImportError as e:
    print("recovering from ImportError '%s'" % e)
    import xml.etree.ElementTree as ET

class CommentedTreeBuilder(ET.TreeBuilder):
    def comment(self, data):
        self.start(ET.Comment, {})
        self.data(data)
        self.end(ET.Comment)


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

cfg_file = './scenario/sumo.cfg.xml'
edges_file = './scenario/edges.xml'
routes_file = './scenario/routes.rou.xml'
nodes_file = './scenario/nodes.xml'
additionals_file = './scenario/lanemeanoutput.add.xml'

parser = argparse.ArgumentParser()
parser.add_argument('-nv', '--num_veh', type=int, default=5)
parser.add_argument('-nl', '--num_lanes', type=int, default=2)
parser.add_argument('-nz', '--num_zones', type=int, default=1)
parser.add_argument('-ls', '--lane_size', type=float, default=1000)
parser.add_argument('-nb', '--num_blockages', type=int, default=0)
parser.add_argument('-bls', '--block_sizes', type=str, default='300')
parser.add_argument('-blp', '--block_pos', type=str, default='750')
parser.add_argument('-bll', '--block_lanes', type=str, default='0')
parser.add_argument('-bdur', '--block_duration', type=str, default='END')
parser.add_argument('-puf', '--pf_update_freq', type=int, default=1)
parser.add_argument('-ec', '--evaporation', type=float, default=0.1)
parser.add_argument('-df', '--diffusion', type=float, default=1.0)
parser.add_argument('--mixed_blockage_training', action='store_true')
parser.add_argument('--nopolicy', action='store_true')


args = parser.parse_args()

print(args.num_zones, 'zones')

# print([float(x) for x in args.block_sizes.split(',')])
# lbs_string = ';'.join([str(x) for x in args.block_sizes])
# lbp_string = ';'.join([str(x) for x in args.block_pos])

# if args.num_blockages > 0:
#     constructed_name = 'nv-{}_nl-{}_nz-{}_ls-{}_nb-{}_bls-{}_blp-{}_bll-{}_bdur-{}_puf-{}_ec-{}_df-{}'.format(args.num_veh, args.num_lanes, args.num_zones, args.lane_size, args.num_blockages, args.block_sizes, args.block_pos, args.block_lanes, args.block_duration, args.pf_update_freq, args.evaporation, args.diffusion)
# else:
if args.mixed_blockage_training:
    constructed_name = 'mixed_blockage_nv-{}_nl-{}_nz-{}_ls-{}_nb-{}_puf-{}_ec-{}_df-{}'.format(args.num_veh, args.num_lanes, args.num_zones, args.lane_size, args.num_blockages, args.pf_update_freq, args.evaporation, args.diffusion)
else:
    constructed_name = 'nv-{}_nl-{}_nz-{}_ls-{}_nb-{}_puf-{}_ec-{}_df-{}'.format(args.num_veh, args.num_lanes, args.num_zones, args.lane_size, args.num_blockages, args.pf_update_freq, args.evaporation, args.diffusion)

nz = int(args.num_zones)
ls = int(args.lane_size)
sz = ls//nz

xml_parser = ET.XMLParser(target=CommentedTreeBuilder())
tree = ET.parse(cfg_file, parser=xml_parser)
        
for child in tree.find('output').iter():
    if isinstance(child.tag, str):
        if child.tag == 'output-prefix':
            child.set('value', 'TIME')
        elif child.tag.endswith('-output'):
            data_dir = os.path.join(*['data', constructed_name + '_' + child.tag])
            data_path = '../{}/{}.xml'.format(data_dir, child.tag)
            os.makedirs(data_dir, exist_ok = True)

            child.set('value', data_path)
        
tree.write(cfg_file,encoding='UTF-8',xml_declaration=True)    

xml_parser = ET.XMLParser(target=CommentedTreeBuilder())
tree = ET.parse(edges_file, parser=xml_parser)
for child in tree.find('edge').iter():
    if 'id' in child.attrib:
        if child.attrib['id'] == '1f2':
            child.set('numLanes', str(args.num_lanes))
tree.write(edges_file,encoding='UTF-8',xml_declaration=True)


xml_parser = ET.XMLParser(target=CommentedTreeBuilder())
tree = ET.parse(routes_file, parser=xml_parser)

for child in tree.findall('vType'):
    if child.attrib['id'] == 'veh':
        if args.nopolicy:
            child.set('lcSpeedGain', str(1))
            child.set('lcStrategic', str(1))
            child.set('lcCooperative', str(1))
        else:
            child.set('lcSpeedGain', str(0))
            child.set('lcStrategic', str(-1))
            child.set('lcCooperative', str(1))

for child in tree.find('flow').iter():
    if child.attrib['id'] == 'f':
        child.set('number', str(args.num_veh))
tree.write(routes_file,encoding='UTF-8',xml_declaration=True)

xml_parser = ET.XMLParser(target=CommentedTreeBuilder())
tree = ET.parse(nodes_file, parser=xml_parser)
for child in tree.getroot():
    if child.tag == 'node':
        if child.attrib['id'] == '1':
            child.set('x', str(0.0))
        elif child.attrib['id'] == '2':
            child.set('x', str(args.lane_size))
tree.write(nodes_file,encoding='UTF-8',xml_declaration=True)

xml_parser = ET.XMLParser(target=CommentedTreeBuilder())
tree = ET.parse(additionals_file, parser=xml_parser)
root = tree.getroot()
list_of_children_to_be_removed = []
for child in root.iter():
    if 'poly' in child.tag:
        list_of_children_to_be_removed.append(child)

for child in list_of_children_to_be_removed:
    root.remove(child)

for i,j in zip(range(1, nz+1), range(0,ls,sz)):
    poly_id = 'zone_{}'.format(i)
    poly_type = 'zone'
    angle = '0.0'
    color = '0,0,255'
    shape = '{},5.0 {},-10.0'.format(float(j),float(j))
    
    lane = 0
    pos = j
    posLat = 0
    child = ET.Element('poly',{'id': poly_id, 'shape': shape, 'type': poly_type, 'color': color, 'layer': '999.0', 'angle': angle, 'lineWidth': '1', })
    
    root.append(child)

indent(root)

tree.write(additionals_file,encoding='UTF-8',xml_declaration=True)