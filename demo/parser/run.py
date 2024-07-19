from FastGaussianPuff import PuffParser

def main():
  p = PuffParser('./dlq.in')
  p.run_exp() # output will appear in the out/ directory

if __name__ == '__main__':
  main()