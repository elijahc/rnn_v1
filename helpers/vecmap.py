"""
This is the VecMap module which supplies a class for managing a set of vectors.
These vectors are mapped to a unique int ID

"""
class VecMap():
    def __init__(self):
        """
        Initialize vector map object
        >>> vmap = VecMap(); vmap.next_key
        0
        >>> vmap.vec_map
        {}
        >>> vmap.vec_set
        set()
        """
        self.vec_set = set()
        self.vec_list = []
        self.vec_map = {}
        self.next_key=0

    def add_tuple(self,tup):
        """
        Add tuple Vector to set

        >>> vmap = VecMap(); vmap.add_tuple((1,2,3))
        0
        >>> vmap.add_tuple((3,4,7))
        1
        """
        if tup not in self.vec_set:
            self.vec_map[tup] = self.next_key
            self.next_key = self.next_key+1
            self.vec_list.extend([tup])
            self.vec_set.add(tup)

        return self.id(tup)

    def add(self,vec):
        """
        Add Vector to set

        >>> vmap = VecMap(); tup = (1,2,3); vmap.add(tup)
        0
        >>> vmap.add(tup)
        1
        """

        if type(vec) == tuple:
            self.add_tuple((self,vec))
        elif type(vec) == numpy.ndarray:
            if len(np.shape(vec)) > 1:
                # Throw an error about needing a 1 dimensional vector
                print('Vector is not 1 dimensional')
            else:
                self.add_tuple( self,tuple( vec.tolist() ) )

    def id(self,vec):
        return self.vec_map[vec]

if __name__ == '__main__':
    import doctest
    doctest.testmod()

