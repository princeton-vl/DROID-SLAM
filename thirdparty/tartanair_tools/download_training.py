from os import system, mkdir
import argparse
from os.path import isdir, isfile

def get_args():
    parser = argparse.ArgumentParser(description='TartanAir')

    parser.add_argument('--output-dir', default='./',
                        help='root directory for downloaded files')

    parser.add_argument('--rgb', action='store_true', default=False,
                        help='download rgb image')

    parser.add_argument('--depth', action='store_true', default=False,
                        help='download depth image')

    parser.add_argument('--flow', action='store_true', default=False,
                        help='download optical flow')

    parser.add_argument('--seg', action='store_true', default=False,
                        help='download segmentation image')

    parser.add_argument('--only-easy', action='store_true', default=False,
                        help='download only easy trajectories')

    parser.add_argument('--only-hard', action='store_true', default=False,
                        help='download only hard trajectories')

    parser.add_argument('--only-left', action='store_true', default=False,
                        help='download only left camera')

    parser.add_argument('--only-right', action='store_true', default=False,
                        help='download only right camera')

    parser.add_argument('--only-flow', action='store_true', default=False,
                        help='download only optical flow wo/ mask')

    parser.add_argument('--only-mask', action='store_true', default=False,
                        help='download only mask wo/ flow')

    parser.add_argument('--azcopy', action='store_true', default=False,
                        help='download the data with AzCopy, which is 10x faster in our test')

    args = parser.parse_args()

    return args

def _help():
    print ''

if __name__ == '__main__':
    args = get_args()

    # output directory
    outdir = args.output_dir
    if not isdir(outdir):
        print('Output dir {} does not exists!'.format(outdir))
        exit()

    # difficulty level
    levellist = ['Easy', 'Hard']
    if args.only_easy:
        levellist = ['Easy']
    if args.only_hard:
        levellist = ['Hard']
    if args.only_easy and args.only_hard:
        print('--only-eazy and --only-hard tags can not be set at the same time!')
        exit()


    # filetype
    typelist = []
    if args.rgb:
        typelist.append('image')
    if args.depth:
        typelist.append('depth')
    if args.seg:
        typelist.append('seg')
    if args.flow:
        typelist.append('flow')
    if len(typelist)==0:
        print('Specify the type of data you want to download by --rgb/depth/seg/flow')
        exit()

    # camera 
    cameralist = ['left', 'right', 'flow', 'mask']
    if args.only_left:
        cameralist.remove('right')
    if args.only_right:
        cameralist.remove('left')
    if args.only_flow:
        cameralist.remove('mask')
    if args.only_mask:
        cameralist.remove('flow')
    if args.only_left and args.only_right:
        print('--only-left and --only-right tags can not be set at the same time!')
        exit()
    if args.only_flow and args.only_mask:
        print('--only-flow and --only-mask tags can not be set at the same time!')
        exit()

    # read all the zip file urls
    with open('download_training_zipfiles.txt') as f:
        lines = f.readlines()
    ziplist = [ll.strip() for ll in lines if ll.strip().endswith('.zip')]

    downloadlist = []
    for zipfile in ziplist:
        zf = zipfile.split('/')
        filename = zf[-1]
        difflevel = zf[-2]

        # image/depth/seg/flow
        filetype = filename.split('_')[0] 
        # left/right/flow/mask
        cameratype = filename.split('.')[0].split('_')[-1]
        
        if (difflevel in levellist) and (filetype in typelist) and (cameratype in cameralist):
            downloadlist.append(zipfile) 

    if len(downloadlist)==0:
        print('No file meets the condition!')
        exit()

    print('{} files are going to be downloaded...'.format(len(downloadlist)))
    for fileurl in downloadlist:
        print fileurl

    for fileurl in downloadlist:
        zf = fileurl.split('/')
        filename = zf[-1]
        difflevel = zf[-2]
        envname = zf[-3]

        envfolder = outdir + '/' + envname
        if not isdir(envfolder):
            mkdir(envfolder)
            print('Created a new env folder {}..'.format(envfolder))
        # else: 
        #     print('Env folder {} already exists..'.format(envfolder))

        levelfolder = envfolder + '/' + difflevel
        if not isdir(levelfolder):
            mkdir(levelfolder)
            print('  Created a new level folder {}..'.format(levelfolder))
        # else: 
        #     print('Level folder {} already exists..'.format(levelfolder))

        targetfile = levelfolder + '/' + filename
        if isfile(targetfile):
            print('Target file {} already exists..'.format(targetfile))
            exit()

        if args.azcopy:
            cmd = 'azcopy copy ' + fileurl + ' ' + targetfile 
        else:
            cmd = 'wget -r -O ' + targetfile + ' ' + fileurl
        print cmd
        system(cmd)