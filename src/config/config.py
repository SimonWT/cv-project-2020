if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        sys.argv.append('defaults.yaml')
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)