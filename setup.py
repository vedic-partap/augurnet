from distutils.core import setup
setup(
  name = 'augur',         
  packages = ['augur'],   
  version = '0.1',      
  license='MIT',        
  description = 'Network Traffic Predictions. We develop the product with the mindset of ease. The predictions are done by using the temporal history of the events from the multiple host. The model take into account the effect of multiple host (network) on user’s traffic and contention between different hosts.',   
  author = 'Vedic Partap',                   
  author_email = 'me@vedicpartap.com',      
  url = 'https://github.com/vedic-partap/augur',   
  download_url = 'https://github.com/vedic-partap/augur/archive/v_01.tar.gz',    
  keywords = ['networ', 'reinforcement-learning', 'predictions', 'generative models'],   
  install_requires=[            
        'numpy>=1.17.2,',
        'torch>=1.3.1',
        'matplotlib>=3.1.1',
        'seaborn>=0.9.0',
        'scikit-learn>=0.21.3'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
