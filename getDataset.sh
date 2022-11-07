curl "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/The%20Eiffel%20Tower.npy" --output 'src/train/eiffel.npy'
curl "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy" --output 'src/train/airplane.npy'
curl "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/banana.npy" --output 'src/train/banana.npy'
curl "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/bee.npy" --output 'src/train/bee.npy'
curl "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/bicycle.npy" --output 'src/train/bicycle.npy'
curl "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/bulldozer.npy" --output 'src/train/bulldozer.npy'

echo 'Dataset Download Complete.'
