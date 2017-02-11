# cardiovascular_toolkit
Master thesis of Zihan Chen, ECE, University of Toronto

<!DOCTYPE HTML>
<html>

<head>
    <title>H H Hash (Final Report)</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <!--[if lte IE 8]><script src="assets/js/ie/html5shiv.js"></script><![endif]-->
    <link rel="stylesheet" href="assets/css/main.css" />
    <!--[if lte IE 8]><link rel="stylesheet" href="assets/css/ie8.css" /><![endif]-->
    <!--[if lte IE 9]><link rel="stylesheet" href="assets/css/ie9.css" /><![endif]-->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
</head>

<body>

    <!-- Header -->
    <section id="header">
        <header class="major">
            <h1>H H HASH</h1>
            <p style="font-weight: 700"><span>Final Report</span></p>
            <p style="margin-top: 1.8em; font-weight: 500">Norman Ponte (nponte) &nbsp;&nbsp; Yiming Zong (yzong)</p>

            <p style="margin-top: 3.5em;">
                <span style="font-weight: 300; text-transform:none;"><a href="./checkpoint.html">Click here</a> for our proposal &amp; checkpoint writeup.<br />
                <span style="font-weight: 300; text-transform:none;">Check out our <a href="/assets/hhhash-final.zip">source code and benchmark data</a> and <a href="/assets/final-presentation.pdf">final presentation</a>.
                </span>
            </p>
        </header>
    </section>

    <!-- One -->

    <section id="one" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>Summary</h2>
                </header>
                <p>The goal of our project is to implement a GPU-based hash table that uses two-level cuckoo hashing, and we compare its performance against fine-tuned, CPU-based hash table implementation. We built HH Hash from scratch in order to have full control over GPU memory management, hash function, and bucket-rebalancing. And, we benchmarked the performance of the hash table with batches of insertions, lookups, and deletions, and measured the scalability of our design. Our source code can be compiled and run on <code>latedays</code> cluster (especially the Tesla and Titanx workers).</p>
                <p>In this writeup, we first describe how two-level cuckoo hashing works and demonstrate how we parallelized the algorithm on TitanX GPU. Then, we present the benchmark result of our hash table, compare it with CPU-based hash table (NBDS), and eventaully conclude that HH Hash gives great performance improvements for huge, batchable workloads, while traditional CPU-based hash tables work better for smaller datasets and individual operations.</p>
            </div>
            <a href="#two" class="goto-next scrolly">Next</a>
        </div>
    </section>

    <!-- Second One -->
    <section id="two" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>Background</h2>
                </header>
                <p>The data structure that we try to parallelize in this project is a hash table. A hash table manages a collection of key-value pairs by supporting the three operations: <code>insert(k,v)</code>, <code>lookup(k)</code>, and <code>delete(k)</code>. The keys and values can be of arbitrary types as long as the key is hashable, and the advantage of a hash table over, say, binary search on a sorted list is that hash table allows all three operations to be completed in constant-time (on average). In fact, hash table is but an abstraction, and it can be implemented using different techniques, including linear/quadratic probing, separate chaining, and cuckoo hashing. And, for this project we focus on parallelizing cuckoo hashing on a GPU because the problem is neither trivial (topic of recent PhD dissertation) nor widely-solved (no public source code available).</p>

                <p> $m$-Cuckoo hashing is an open-addressing hashing strategy that maps a key to one of $m$ hash value candidates, i.e. $\{ h_i(k) \mid i\in[m] \}$. Here is what happens when one attempts to insert <code>(k,v)</code> into the hash table: the hash table checks if the slot $h_0(k)$ is already occupied -- if not, we place <code>(k,v)</code> there and succeed; otherwise, we replace the entry there with <code>(k,v)</code>, and then attempt to hash the original key-value pair with an alternative hash function, say, $h_1(k')$. Notice that this is a highly sequential process, since the behavior at each step is determined by whether the key-value pair in the previous step was placed in an empty spot, and, if not, which key-value pair was evicted. The <a href="https://en.wikipedia.org/wiki/Cuckoo_hashing">Wikipedia page</a> for Cuckoo Hashing contains several examples for motivated readers.</p>

                <p>The naive sequential algorithm can be inefficient not only due to lack of parallelism but also due to poor cache locality. When some <code>(k,v)</code> is being inserted, it may swap with another key-value pair <i>anywhere</i> on the hash table, since the value of $h_i(k)$ is not bounded. This results in random read/write pattern on the hash table, which may cause significant cache line thrashing. However, we claim that cuckoo hashing can be parallelized in order to help hide the memory latency. Instead of inserting elements one after another, we can perform <i>batch insertions</i>, as follows: firstly, we hash every element to be inserted by its first hashing function, place them in the hash table, while potentially swapping out some existing entries. Then, anything that was swapped out are hashed with the second function, and the same swapping process repeats.</p>

                 <p>On top of $m$-Cuckoo hashing, we also use FKS hashing to augment our hash table into two layers. FKS hashing is a perfect hashing scheme where each bucket with $n$ elements is allocated $n^2$ slots. This assures that the augmented hash table has few collisions at the cost of extra memory usage. More theoretical information is available on its <a href="https://en.wikipedia.org/wiki/Dynamic_perfect_hashing#FKS_Scheme">Wikipedia page</a>.</p>
 
                 <p>As for the target use case of our hash table, we notice that nowadays people often manipuate huge datasets and that fast lookup time is very important. And, one of the most fundamental data structures for this purpose is a hash table. With this in mind we want to develop a hash table that achieves significant speedup with larget datasets. While many fine-tuned CPU-based hash tables are available and commonly used, GPU-based hash tables have only appeared recently in PhD-level work. Therefore, we would like to explore if we can utilize the massive parallel computing power of GPU in order to achieve significant improvements in read/write/delete throughput.</p>

                 <p>
                    For this project, parallelism is a means to achieve an end. In order to achieve high throughput improvements, we need parallelism for higher level of concurrency and rebalancing for more even distribution of the workload among the GPU blocks. Meanwhile, we still need to ensure that our hash table has all the properties of a CPU-based one (e.g. atomicity) by applying synchronization primitives. As learnt in the course, synchronization overhead can be significant for a highly-contended shared resource, and thus we debate and select the most efficient synchronization primitives for our hash table. 
                </p>
            </div>
            <a href="#three" class="goto-next scrolly">Next</a>
        </div>
    </section>

    <!-- Third One -->
    <section id="third" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>Approach</h2>
                    </header>
                    <p>While designing our approach, we aimed to tackle every weakness of naive cuckoo hashing individually. As argued in the previous section, one of the main issues is that cuckoo hashing exhibits little natural locality. Thus, we break the original flat hash table into different buckets (with FKS hashing). By doing so, we reduce the range in which memory reads/writes jump around as key-values pairs are evicted and substituted in, thereby improving cache locality. Another issue we mentioned is that different steps in cuckoo hashing are dependent. Unfortunately, there is little we can do about that, but we can compensate it by performing hash table operations <i>by batches</i>. The remainder of the section mainly discusses how we parallelized the bulk operations on the hash table.</p>

                    <p>Overall, our hash table has a two-layer structure, with FKS hashing on top of individual cuckoo hash tables. Upon a <i>batch insert</i> operation, each input element first finds a FKS bucket and then performs cuckoo hashing with other elements in the same bucket. If no viable layout for the bucket is found, we rehash the entire bucket after providing the bucket with new hash functions (by generating new random seeds). In general, we parallelize over the entries in batch operations given by the user, and our final product is a library that can be used as long as a GPU exists. The majority of our hash table code is written by using the thrust library for CUDA. The reason for our decision is that we wanted to rapidly prototype through different algorithms and implementations in order to determine which one to select. While the rough ideas behind our algorithm have been described in the PhD thesis by Dan Alcantara, our implementation and tuning is entirely original, and thus we wanted to be able to switch rapidly in case something we attempt was headed in the wrong direction.</p>

                    <p>Our batch workloads are broken into blocks which are then processed by device kernels that modify the underlying data structures of our hash map. While cuckoo hashing is intrinsically sequential, we were able to parallelize it in two ways:
                    <ul>
                    <li>Our first implentation pushes entries into the hash map "optimistically" regardless of the state of previous attempts. This approach has more parallelism, but the cuckoo hashing is not performed in the correct order ($h_0(k), h_1(k),\cdots$), thereby causing a lot of spurious collisons and bucket rehashing. We later noticed that if we randomized the cuckoo hashing order we could do more operations in parallel, and we though that if we allowed more cuckoo hashing iterations then the hash table would converge to a layout that fit all the inputs. However, we were being too optimistic, and in fact "randomized cuckoo hashing" seemed to stall the progress with a high chance, such that very few buckets could find a satisfactory layout.<br /></li><br />

                    <li>Our second attempt was that at first we wanted the buckets to be worked upon in sequence, in order to exploit cache locality. By keeping the bucket size low, we hoped that the bucket would fit inside the cache, and the shuffling caused by cuckoo hashing would experience a speedup. However, this speedup turned out to be overpowered by the extra cost of sorting the input entries by their bucket number, because NVIDIA Visual Profiler (<code>nvvp</code>) indicated that our system was taking 80% of the time in the GPU merge-sorting the array. This meant we were losing time in general. Meanwhile, we also noticed that we needed to have a large bucket size; otherwise the hash value space would be compressed into $[0, \texttt{BUCKET_SIZE})$, thereby increasing the rate of "false" hash value collisions sharply.</li>
                    </ul>

                    <p>After deciding on our final implementation we had many parameters to fine-tune and optimize, including how full the buckets normally are, when to make more buckets, when to coalesce old buckets, how many loop iterations we allow cuckoo hashing to run through before rehashing the bucket with new hash functions. Our final code contains the parameter set that gave the most optimal result for our test traces, but we understand that the optimal parameters may vary depending on the usage characteristics.</p>
                </header>
           </div>
            <a href="#four" class="goto-next scrolly">Next</a>
        </div>
    </section>

 
    <!-- Four One -->
    <section id="four" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>Challenges</h2>
                </header>
                <p>Speed is the name of the game. Very intergral to our design is the speed of operations and the idea that what we are working with the largest datasets. The main challenge for us is to make sure that our design works for different workloads and access patterns. Also, in order to reduce the amount of CUDA code that we write manually, we need to learn how to use <a href="http://docs.nvidia.com/cuda/thrust/">Thrust library</a>.</p>
                <p>Meanwhile, in order to gain enough background knowledge on efficient implementations of hash table, we need to read a lot of papers and dissertations, which are listed in the references section. We mainly focus on papers about open addressing (especially cuckoo hashing) and lock-free hash table implemetations; meanwhile, we avoided browsing code repos on Github such that we could come up with our own design entirely without being influenced by existing solutions.</p>
                <p>As we read more about cuckoo hashing, we noticed that it is intrinsically a sequential algorithm. More specifically, each key can be potentially hashed into multiple locations, and collisions are resolved by forced eviction of the previous key. Therefore, it can be tricky to parallelize different conflicting "key insertion" requests, each of which having strong side-effects.</p>
            </div>
            <a href="#five" class="goto-next scrolly">Next</a>
        </div>
    </section>

    <!-- Fifth One -->
    <section id="five" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>Results</h2>
                    </header>
                    <p> There are two main metrics that are used to measure the performance of our hash tables: operation throughput (measured in number of ops per second), and memory footprint. Most important to us is the former, since it is what the users of our library care about. In order to test the correctness of our implementation and tune parameters, we created operation traces and ran construction, insertion, deletion, and lookup on the hash table. In this section, the key-value trace we use in this section is $\langle (i+1, 3i+5) \mid i\in\{0,\cdots,\texttt{len}-1\}\rangle$ because it is easy to generate and because the hash function would scramble the bits of the keys anyways. Following are the performance charts we obtained, along with our interpretations:

                    <p>Firstly, when we ran insertion and deletion on the hash table, we noticed that they have very similar performance characteristics, as shown in the chart below. As the data size increases, we observe a general increase in operation throughput because higher amount of data is able to utilize the parallelism more fully, and because the proportion of useful work increases. Also, the performance curve for insertions has more turbulence because the bucket rehashing operation happens randomly and is time-consuming.</p>

                    <div>
                    <a href="https://plot.ly/~alsdkfj/19/" target="_blank" title="Insertion / Deletion Throughput vs. Data Size" style="display: block; text-align: center;"><img src="https://plot.ly/~alsdkfj/19.png" alt="Insertion / Deletion Throughput vs. Data Size" style="max-width: 100%;width: 1258px;"  width="1258" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
                    <script data-plotly="alsdkfj:19"  src="https://plot.ly/embed.js" async></script>
                    </div>


                    <p><br />Needless to say, lookup speed is also a crucial benchmark of our hash table since lookup is the most common use case in real-world scenarios. The benchmark of lookup speed can be found below. Notice that both axes are log-scaled, and our hash table scales very well up to the point where GPU memory runs out. This can be explained by the fact that cuckoo hashing has $\mathcal{O}(1)$ lookup time, whose implication to lookup performance is very noticable for large lookup requests.</p>
                    <div>
                    <a href="https://plot.ly/~alsdkfj/20/" target="_blank" title="Lookup Throughput vs. Data Size" style="display: block; text-align: center;"><img src="https://plot.ly/~alsdkfj/20.png" alt="Lookup Throughput vs. Data Size" style="max-width: 100%;width: 1463px;"  width="1463" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
                    <script data-plotly="alsdkfj:20"  src="https://plot.ly/embed.js" async></script>
                    </div>

                    <p><br />An interesting parameter in our hash table is <i>bucket size (load)</i>. Having a smaller bucket size would generate extra memory overhead (36 bytes per buket), and it would also narrow down the hash value space and cause more collisions and bucket-rehashing, which are heavily time-consuming; however, it could potentially fit into GPU memory cache, thereby reducing memory latency. The goal of our empirical analysis is to determine if the cache locality is worth the extra rehashings. Following is the chart with our benchmark result, and it turns out that we are better off with larger buckets and ignore cache locality.</p>

                    <div>
                    <a href="https://plot.ly/~alsdkfj/21/" target="_blank" title="Insertion / Deletion Throughput vs. Bucket Load" style="display: block; text-align: center;"><img src="https://plot.ly/~alsdkfj/21.png" alt="Insertion / Deletion Throughput vs. Bucket Load" style="max-width: 100%;width: 1463px;"  width="1463" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
                    <script data-plotly="alsdkfj:21"  src="https://plot.ly/embed.js" async></script>
                    </div>

                    <p><br />Another parameter that matters a lot is $\texttt{REG_LOAD}$, which describes the load of a bucket under "normal" circumstance. This can be classified as a time-memory tradeoff, as a higher load factor saves memory but makes a more crowded hash table, and thus more cuckoo hashing iterations are required in order to find a satisfactory layout, and there would also be a higher chance for rehashing. The reverse goes for a low load-factor -- we are gaining performance wins at the cost of memory overhead. Meanwhile, the load factor should not be too small; otherwise there would be near-empty buckets everywhere, incurring spurious memory &amp; processing overhead. Following is the chart where we quantify the effect of load factor on insertion throughput, and it seems that 0.35 is the sweet spot for us:</p>

                    <div>
                    <a href="https://plot.ly/~alsdkfj/22/" target="_blank" title="Insertion Throughput vs. REG_LOAD" style="display: block; text-align: center;"><img src="https://plot.ly/~alsdkfj/22.png" alt="Insertion Throughput vs. REG_LOAD" style="max-width: 100%;width: 1463px;"  width="1463" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
                    <script data-plotly="alsdkfj:22"  src="https://plot.ly/embed.js" async></script>
                    </div>

                    <p><br />Eventually, we want to see how well we perform in the real-world by comparing against other hash-table implementation. The candidate we picked is NBDS Hash Table, which is considered a highly-performant, lock-free, concurrent hash table. When we compared our insertion performance against theirs (running on 48 threads on <code>latedays</code>), we noticed that we performed poorly with small data size because the time taken to perform <code>cudaMalloc</code> and transfer data between host and data totally overpowers the actual calculation time (more than 20-to-1), and because small dataset does not utilize GPU's computational power effectively. However, as the size of data goes up to $2^{24}$, NBDS Hash Table runs out of memory, while our GPU implementation achieves significant performance win and seemes to be able to scale further. This directly implies that our HH Hash is most efficient for <i>huge</i> datasets while inferior for smaller ones.

                    <div>
                    <a href="https://plot.ly/~alsdkfj/24/" target="_blank" title="Insertion Performance of HH Hash vs. NBDS Hash Table" style="display: block; text-align: center;"><img src="https://plot.ly/~alsdkfj/24.png" alt="Insertion Performance of HH Hash vs. NBDS Hash Table" style="max-width: 100%;width: 1463px;"  width="1463" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
                    <script data-plotly="alsdkfj:24"  src="https://plot.ly/embed.js" async></script>
                    </div>
                    
                    <p><br /> As we reasoned through our performance, we found two main factors that limit our speedup. The first is the more obvious: we run our commands on large input arrays and therefore have to copy over the input array into device code in order to run our hashing algorithms on it. With very large input size, we are memory-bound, as confirmed by <code>nvcc</code>, where host-to-device and device-to-host memory transfer takes up a significant portion of the time. The similar issue occurs when we copy large chunks of result from device memory to host memory while returning from <code>lookup(k*, v*)</code>. The second major slowdown comes from divergence: in cuckoo hashing when the first bucket hash fails, all the blocks move onto the second bucket hash regardless of whether they succeeded in the first iteration. In our testing, we found out that ~70% of our elements found a spot in the first iteration, and therefore 70% of the computational resource is wasted by waiting on the remaining 30% that had not found a block. The same happens again before we finally converge the blocks.</p>
                   <p>Apart from the major factors above, a minor factor is that we used atomic increments and decrements to variables in different places, which could potentially slow down the system with contentions and bus traffic. However, this is necessary in order to guarantee the correctness of parallelized cuckoo hashing, and an atomic counter is the most light-weight synchronization solution possible that we found.</p>
                   <p>Throughout our development cycle, we used <code>nvprof</code> and <code>nvvp</code> extensively, and were able to find many performance issues including merge-sorting by bucket ID taking up to 83% of GPU time. After repeated optimizations by examining the kernels that take the longest run-time, give performance warnings on <code>nvvp</code> (e.g. kernel has low concurrency), or otherwise have suspecious patterns on the timeline, we have brought the calculations on GPU related to hashing to an efficient state. Since the utilization of GPU as reported by <code>nvvp</code> is consistently high, we believe that our workload is suitable for a GPU. However, in initialization/insertion where we create buckets and combine key-value pairs into a single long vector, the performance issue is huge (e.g. Line 1017 and 1018 in <code>GPUHash.cu</code>), and we eventually believe that the performance can be much better if the chunks of the vectors were distributed across the different blocks. However, our realization didn't come until pretty late in the project, and we didn't want to break our hash table by changing the fundamental data structure. We believe that by splitting the contiguous huge memory allocation into smaller separate ones, the performance of our code (especially the last diagram above) would look much more impressive.</p>
                   <p>As an ending note, our HH Hash starts to give out-of-memory error on TitanX (with 12GB memory) when there are more than 100 million entries in the hash table. This implies that our memory overhead for huge tables (which is what we care about) is around 15x. This might seem large, but considering that NBDS hash ran out of 16GB RAM for 32 million entries, our memory usage is acceptable. We leave this at the last since it is less significant compare to the other performance-related results.</p>
                </header>
          </div>
            <a href="#six" class="goto-next scrolly">Next</a>
        </div>
    </section>

    <!-- Six One -->
    <section id="six" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>References</h2>
                </header>
                <p>The original idea of the project comes from a pair of Berkeley assignments (<a href="http://www-inst.eecs.berkeley.edu/~cs162/fa12/phase3.html">1</a>, <a href="http://www-inst.eecs.berkeley.edu/~cs162/fa12/phase4.html">2</a>) on distributed hash table over multiple nodes. Since the assignments are more related to distributed systems (e.g. loda-balancing, fault-tolerance), for this project we decided to make a hash table on a single node, but based on a GPU. This is highly relevant to what has been taught in the class, as the project require us to write CUDA code and design efficient parallelism and synchronization.</p>

                <p>At the beginning, our original idea was to use <a href="http://legion.stanford.edu/">Legion</a> to implement a distributed hash table that allows automatic load-balancing. Even though Legion makes it easier to manipulate data structures in shared memory, we had major issues integrating it in our implementation, and thus we decided to abandon the idea and write the GPU-based hash table from scratch. One of the main reasons we wanted to use Legion was only because our original idea was a distributed GPU hash table. However as we were writting the distributed code we realized that the overhead for a distributed GPU hash table makes it unworthy. Since GPU has less memory then the CPU then a CPU implementation with no communication overhead would outdo most implementations we could resonaly come up with. </p>

                <p>While reading about state-of-the-art implementations of hash tables based on GPU, we came across the PhD dissertation by Dan Alcantara, titled <a href="http://idav.ucdavis.edu/~dfalcant/downloads/dissertation.pdf">Efficient Hash Tables on the GPU</a>, which gives a comprehensive discussion about hashing (including open-addressing and cuckoo-hashing) brief description of GPU-based implementations. From the dissertation, we mainly learned the mechanism of cuckoo hashing, and how GPU allows massive parallelism with hash tables. </p>
                
                <p>While learning about different hash functions and their GPU implementations, we came across the paper titled <a href="http://research.microsoft.com/en-us/um/people/hoppe/perfecthash.pdf">Perfect Spatial Hashing</a> by Lefebvre &amp; Hoppe from Microsoft Research.
                </p>
            </div>
            <a href="#seven" class="goto-next scrolly">Next</a>
        </div>
    </section>

    <!-- sven One -->
    <section id="seven" class="main special">
        <div class="container">
            <span class="image fit primary"><img src="images/test1.jpg" alt="" /></span>
            <div class="content">
                <header class="major">
                    <h2>Future Work</h2>
                </header>
                <p>Our strech goal is to implement a hybrid hash table. In our literature review, we realized that GPU-based hash table is bound by GPU memory; therefore, if we need to store billions of key-value pairs, GPU memory may run out, and our project would no longer be useful. However, if we can use CPU-based hash map as a "spillover area" and only keep the "heavy-hitters" in GPU, we would have a powerful hybrid system that gives much better capacity, where the GPU-based hash table is essentially a cache for CPU-based hash table. However, this is non-trivial and requires an efficient algorithms for keeping track of the heavy hitters. We believe that it could potentially be a very interesting independent study project.</p>
            </div>
            <div class="content">
                <header class="major">
                    <h3>Work Distribution</h3>
                </header>
                <p>Equal work was performed by both project members.</p>
            </div>
        </div>
        <footer>
            <p> 15-418 Spring 2016 Final Project </p>
            <ul class="copyright" style="list-style-type: none;">
                <li>Design: <a href="http://html5up.net">HTML5 UP</a>
                </li>
                <li>Demo Images: <a href="http://unsplash.com">Unsplash</a>
                </li>
            </ul>
        </footer>
    </section>

     <!-- Scripts -->
    <script src="assets/js/jquery.min.js"></script>
    <script src="assets/js/jquery.scrollex.min.js"></script>
    <script src="assets/js/jquery.scrolly.min.js"></script>
    <script src="assets/js/skel.min.js"></script>
    <script src="assets/js/util.js"></script>
    <!--[if lte IE 8]><script src="assets/js/ie/respond.min.js"></script><![endif]-->
    <script src="assets/js/main.js"></script>

</body>

</html>
