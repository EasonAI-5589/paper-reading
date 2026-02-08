# BUILDING SETS, CHOW RINGS, AND THEIR HILBERT SERIES

CHRISTOPHER EUR, LUIS FERRONI, JACOB P. MATHERNE, ROBERTO PAGARIA, AND LORENZO VECCHI

Abstract. We establish formulas for the Hilbert series of the Feichtner–Yuzvinsky Chow ring of a polymatroid using arbitrary building sets. For braid matroids and minimal building sets, our results produce new formulas for the Poincar´e polynomial of the moduli space $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ of pointed stable rational curves, and recover several previous results by Keel, Getzler, Manin, and Aluffi–Marcolli–Nascimento. We also use our methods to produce examples of matroids and building sets for which the corresponding Chow ring has Hilbert series with non-log-concave coefficients. This contrasts with the real-rootedness and log-concavity conjectures of Ferroni–Schr¨oter for matroids with maximal building sets, and of Aluffi–Chen–Marcolli for braid matroids with minimal building sets.

# 1. Introduction

A polymatroid M is a pair $( E , \operatorname { r k } )$ consisting of a finite set $E$ , called the ground set, and a function rk: $2 ^ { E } \to \mathbb { Z } _ { \geq 0 }$ , called the rank function, satisfying the following properties:

• $\operatorname { r k } ( \varnothing ) = 0$ , • if $A _ { 1 } \subseteq A _ { 2 }$ , then $\operatorname { r k } ( A _ { 1 } ) \leq \operatorname { r k } ( A _ { 2 } )$ , and $\bullet$ for all $A _ { 1 } , A _ { 2 } \subseteq E$ , one has

$$
\operatorname { r k } ( A _ { 1 } ) + \operatorname { r k } ( A _ { 2 } ) \geq \operatorname { r k } ( A _ { 1 } \cup A _ { 2 } ) + \operatorname { r k } ( A _ { 1 } \cap A _ { 2 } ) .
$$

If furthermore $\operatorname { r k } ( A ) \leq | A |$ for every $A \subseteq E$ , then $\mathsf { M }$ is a matroid. A flat of $\mathsf { M }$ is a subset $F \subseteq E$ such that $\operatorname { r k } ( F \cup \{ e \} ) > \operatorname { r k } ( F )$ for all $e \in E \setminus F$ . The set of all flats of $\mathsf { M }$ ordered by inclusion is a lattice, denoted $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . We will always assume that a polymatroid M is loopless, i.e. that $\emptyset \in { \mathcal { L } } ( \mathbb { M } )$ . We refer to [Wel76, Sch03] for detailed treatments of polymatroids.

A subset ${ \mathcal { G } } \subseteq { \mathcal { L } } ( { \mathsf { M } } )$ of nonempty flats is a building set if it satisfies a factorability condition (see Definition 2.1). To any polymatroid $\mathsf { M }$ and building set $\mathcal { G }$ , Feichtner and Yuzvinsky [FY04] associated a Chow ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ (see Definition 2.6), motivated by the geometry of wonderful compactifications of De Concini and Procesi [DCP95]. It is an Artinian graded ring $D ( \mathsf { M } , \mathscr { G } ) = \bigoplus _ { i } D ^ { i } ( \mathsf { M } , \mathscr { G } )$ whose Hilbert series is denoted

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) : = \sum _ { i \geq 0 } \dim \left( D ^ { i } ( \mathsf { M } , \mathcal { G } ) \right) x ^ { i } .
$$

Let us highlight two particular instances of the Chow ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ :

• When M is a matroid, the Chow ring $D ( \mathsf { M } , \mathcal { G } _ { \mathrm { m a x } } )$ with respect to the maximal building set $\mathcal { G } _ { \mathrm { m a x } }$ is the Chow ring of the matroid M as given in [AHK18, Definition 1.3]. It is a central object in the Hodge theory of matroids [AHK18] that resolved the Heron–Rota–Welsh conjecture [Rot71, Her72, Wel76] on the logconcavity of the coefficients of the characteristic polynomial of a matroid.

• When $\mathsf { M }$ is a braid matroid $\mathsf { K } _ { n }$ , i.e. the graphic matroid associated to a complete graph on $n$ vertices, the ring $D ( \mathsf { K } _ { n } , \mathcal { G } _ { \operatorname* { m i n } } )$ with respect to the minimal building set $\mathcal { G } _ { \mathrm { m i n } }$ is isomorphic to the Chow ring of the Deligne–Knudsen–Mumford moduli space $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ of stable rational curves with $( n + 1 )$ marked points.

A myriad of works have studied the Hilbert series $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ in these two cases:

• For matroids with maximal building sets, see [BES24, BHM $^ +$ 22, FS24, FMSV24] for proofs of combinatorial interpretations, valuativity, and general recursions; [Stu24, FMV24] for inequalities for Chow polynomials in the broader context of graded posets; [ANR25, Lia24] for equivariant counterparts; and [HRS21, Hos24, BV25] for results that• For the braid matroid $\mathsf { K } _ { n }$ tain to special cases such as uniform matroids., three different formulas for the Hilbert series $\mathrm { H } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x )$ Marcolli, and Nascimento [AMN24], all via the geometry of the space $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ .

The ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ is known to satisfy a trio of properties known as the K¨ahler package [PP23, Section 4]; see also [ADH23] and [CHL $^ +$ 25]. One of the components of the package, the Hard Lefschetz theorem, implies that the sequence of coefficients of the polynomial $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ is symmetric and unimodal. Under the additional assumption that M is a matroid and $\mathcal { G }$ is the maximal building set, it was conjectured by Ferroni and Schr¨oter in [FS24] that the corresponding Hilbert series are real-rooted polynomials. Real-rootedness is the strongest in the hierarchy of properties depicted in Figure 1 (see [Br¨a15, Bre94, Sta89]). The weaker property of $\gamma$ -positivity was established in this special case [FMSV24].

![](images/9ceb01a44094b9bfc570ef419503895052e7e30208eb6e53e2581e3fb9618581.jpg)  
Figure 1. Hierarchy of properties for palindromic polynomials

In the general setting of arbitrary building sets, all of the properties appearing in Figure 1, except unimodality, fail to be true. The failure of real-rootedness and $\gamma$ -positivity is easily found: when $\mathsf { M }$ is a uniform matroid of rank $k \geq 3$ and $\mathcal { G } = \mathcal { G } _ { \operatorname* { m i n } }$ , one has $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = 1 + x + \cdot \cdot \cdot + x ^ { k - 1 }$ . The failure of log-concavity will be given in Theorem 1.6 (with more details in Section 6). The construction of this example is an application of some recursions we develop that generalize those in [FMSV24] to the case of arbitrary building sets.

1.1. Main results. Our first main results consist of a pair of recursive formulas for the Hilbert series $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ of the Chow ring of an arbitrary building set $\mathcal { G }$ on a polymatroid $\mathsf { M }$ . Let us fix additional notation to state it. The characteristic polynomial of a polymatroid $\mathsf { M }$ , denoted $\chi _ { \mathsf { M } } ( x )$ , is defined as

$$
\chi _ { \mathsf { M } } ( x ) = \sum _ { F \in \mathcal { L } ( \mathsf { M } ) } \mu ( \emptyset , F ) x ^ { \mathrm { r k } ( E ) - \mathrm { r k } ( F ) } ,
$$

where $\mu$ denotes the M¨obius function of the poset $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ (see [Sta12, Chapter 3]). The set of $\mathcal { G }$ -factors of a building set $\mathcal { G }$ on $\mathsf { M }$ , denoted $f ( \mathcal G )$ , consists of all maximal elements in the subset $\mathcal { G }$ of the poset $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . For a flat $F$ of $\mathsf { M }$ , a building set $\mathcal { G }$ naturally induces building sets $\mathcal { G } | _ { F }$ and $\mathcal { G } / F$ on the restriction $\mathsf { M } | _ { F }$ and the contraction $\mathsf { M } / F$ polymatroids, respectively (see Definition 2.3). If $\mathsf { M }$ is the empty polymatroid, we set $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = 1$ .

Theorem 1.1 For every nonempty polymatroid M and every building set $\mathcal { G }$ , the following two recursions hold:

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { E } } } ( x ) = \sum _ { \substack { \emptyset \neq F \in \mathcal { L } ( \mathsf { M } ) } } \frac { - \chi _ { \mathsf { M } | _ { F } } ( x ) } { ( 1 - x ) ^ { | f ( \mathfrak { G } | _ { F } ) | } } \cdot \mathrm { H } _ { \mathsf { M } / F } ^ { \mathsf { \mathcal { G } } / F } ( x ) ,
$$

and

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { G } } } ( x ) = \sum _ { E \neq F \in { \mathcal { L } } ( \mathsf { M } ) } \mathrm { H } _ { \mathsf { M } | _ { F } } ^ { \mathsf { { g } } | _ { F } } ( x ) \cdot \frac { - \chi _ { \mathsf { M } / F } ( x ) } { ( 1 - x ) ^ { | f ( \mathsf { \mathcal { G } } / F ) | } } .
$$

We prove these recursions using the main result of Feichtner and Yuzvinsky in [FY04], which consists of a Gr¨obner basis computation leading to a basis of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ as a $\mathbb { Q }$ -vector space. Theorem 1.1 is equivalent to the statement (Theorem 3.6) that two particular elements in the incidence algebra of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ are inverses. Thus, using the formalism of incidence algebras, we derive a non-recursive formula as a sum over chains of flats.

Corollary 1.2 The Hilbert series of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ equals

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { g } } ( x ) = \sum _ { m \geq 0 } \sum _ { \substack { \emptyset = F _ { 0 } \subset \cdots \subset F _ { m + 1 } = E } } ( - 1 ) ^ { m } \prod _ { i = 1 } ^ { m + 1 } \frac { \chi _ { \mathsf { M } | _ { F _ { i } } / F _ { i - 1 } } ( x ) } { ( 1 - x ) ^ { | f ( \mathcal { G } | _ { F _ { i } } / F _ { i - 1 } ) | } } .
$$

In the preceding statement, the sum runs over all possible chains of flats anchored at the bottom and top elements. We also provide a more efficient way of computing the polynomial $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ , in terms of $\mathcal { G }$ -nested sets (Definition 2.5) that are spanning in the sense that the join of its elements is the ground set $E$ .

Theorem 1.3 The Hilbert series of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ equals

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = \sum _ { S } \prod _ { F \in S } \overline { { \chi } } _ { \mathsf { M } | _ { F } / \operatorname* { s u p } ( S , F ) } ( x ) ,
$$

where the sum runs over all spanning $\mathcal { G }$ -nested sets, and $\operatorname { s u p } ( S , F )$ denotes the join of all the elements in $S \subseteq { \mathcal { G } }$ which are strictly smaller than $F$ .

The formulas in Corollary 1.2 and Theorem 1.3 are useful for computing Hilbert series of rings $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ when the characteristic polynomials of the intervals of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ obey a predictable pattern. A central example is when ${ \textsf { M } } = \mathsf { K } _ { n }$ : in this case, the reduced characteristic polynomials of intervals of flats are products of polynomials of the form $( x - 2 ) ( x - 3 ) \cdots ( x - r )$ . When $\mathcal { G } = \mathcal { G } _ { \operatorname* { m i n } }$ is the minimal building set, the wonderful variety (see Section 3.5) is the moduli space $\mathcal { M } _ { 0 , n + 1 }$ . Our results lead us to a new closed formula for its Poincar´e polynomial.

Theorem 1.4 The Poincar´e polynomial of $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ is

$$
P _ { \overline { { \mathcal M } } _ { 0 , n + 1 } } ( x ) = \sum _ { \lambda \vdash [ n - 1 ] } \frac { ( n - 1 + \ell ( \lambda ) ) ! } { ( n - 1 ) ! } \prod _ { i = 1 } ^ { \ell ( \lambda ) } \frac { \overline { { \chi } } _ { \lambda _ { i } + 1 } ( x ) } { \lambda _ { i } + 1 } ,
$$

where the sum is over set partitions of the set $[ n - 1 ] = \{ 1 , \dots , n - 1 \}$ into $\ell ( \lambda )$ nonempty parts, $\lambda _ { i }$ is the cardinality of the $i$ -th part, and $\overline { { { \chi } } } _ { m } ( x ) : = ( x - 2 ) ( x - 3 ) \cdot \cdot \cdot ( x - m + 1 )$ .

From Theorem 1.3, we also deduce a new proof of a recent result by Aluffi, Marcolli, and Nascimento.

Theorem 1.5 ([AMN24, Theorem 1.1]) The Poincar´e polynomial of $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ satisfies

$$
P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) = ( 1 - x ) ^ { n } \sum _ { k \geq 0 } \sum _ { j \geq 0 } s ( k + n , k + n - j ) S ( k + n - j , k + 1 ) x ^ { k + j } ,
$$

where $s ( n , k )$ and $S ( \boldsymbol n , \boldsymbol k )$ denote, respectively, the signed Stirling numbers of the first kind and the Stirling numbers of the second kind.

Furthermore, from our formulas we can also recover generating functions and recurrences for $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ due to Manin and Getzler (see Proposition 4.5 and Proposition 4.6).

For the Chow ring of a matroid M with the maximal building set $\mathcal { G } _ { \mathrm { m a x } }$ , an open conjecture of Ferroni–Schr¨oter [FS24] posits that the Hilbert series $\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { G } _ { \mathrm { m a x } } } ( x )$ is real-rooted. However, for general building sets this property and the weaker property of $\gamma$ -positivity do not hold. We provide a negative answer to the analogous question for log-concavity.

Theorem 1.6 There exists a matroid M such that $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x )$ has non-log-concave coefficients. For any sufficiently large $n$ , there exists a building set $\mathcal { G }$ on the Boolean matroid $\mathsf { U } _ { n , n }$ such that $\mathrm { H } _ { \mathsf { U } _ { n , n } } ^ { \mathcal { G } } ( x )$ has non-log-concave coefficients.

The claimed examples are provided in Section 6. For the construction of these examples, it is useful to note the underlying geometry (in the realizable case) behind the formulas in Theorem 1.1. This underlying geometry of sequential blow-ups is explained in Section 3.5. For those wishing to avoid the geometry of blow-ups, the combinatorial counterpart (which holds regardless of realizability) is provided by Theorem 5.1, which relates the Hilbert series of Chow rings of different building sets.

# Acknowledgments

We are grateful to the organizers of the Oberwolfach workshop “Arrangements, matroids, and logarithmic vector fields”—this project was initiated at that MFO workshop. The authors also thank Emil Verkama for insightful discussions. Chris Eur was partially supported by the National Science Foundation grant DMS-2246518. Luis Ferroni is a member at the Institute for Advanced Study, funded by the Minerva Research Foundation. Jacob Matherne received support from a Simons Foundation Travel Support for Mathematicians Award MPS-TSM-00007970. Roberto Pagaria is partially supported by PRIN 2022 “ALgebraic and TOPological combinatorics (ALTOP)” CUP J53D23003660006 and by INdAM - GNSAGA Project CUP E53C23001670001.

# 2. Preliminaries

We collect preliminary facts about building sets and their Chow rings.

2.1. Building sets. Let us consider a loopless polymatroid M, and assume that we have some distinguished subset of flats $\mathcal { G } \subseteq \mathcal { L } ( \mathsf { M } )$ . We will write

$$
\mathcal { G } _ { \leq F } : = \{ G \in \mathcal { G } : G \subseteq F \} ,
$$

for each $F \in \mathcal { L } ( \mathsf { M } )$ . We define

f(G) := the set of maximal elements in the subset $\mathcal { G }$ of the poset $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ .

The set $f ( \mathcal G )$ will be referred to as the set of $\mathcal { G }$ -factors.

Definition 2.1 Let $\mathsf { M }$ be a loopless polymatroid, and let $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ be its lattice of flats. A subset ${ \mathcal { G } } \subseteq { \mathcal { L } } ( \mathsf { M } ) \setminus \{ \emptyset \}$ is said to be a (geometric) building set if for any $F \in \mathcal { L } ( \mathsf { M } )$ the morphism of lattices

$$
\varphi _ { F } \colon \prod _ { G \in f ( \mathscr { G } _ { \leq F } ) } [ \emptyset , G ] \to [ \emptyset , F ]
$$

induced by the inclusions is an isomorphism, and the following equality holds:

$$
\operatorname { r k } ( F ) = \sum _ { G \in f ( \mathcal { G } _ { \leq F } ) } \operatorname { r k } ( G ) .
$$

Remark 2.2 We emphasize that we do not require the ground set $E$ to be part of a building set, in contrast to some other works. Our definition of a building set coincides with the original definition by De Concini and Procesi [DCP95].

There are two distinguished building sets. The first, called the maximal building set, consists of all nonempty flats, i.e.,

$$
\mathcal { G } _ { \mathrm { m a x } } : = \mathcal { L } ( \mathsf { M } ) \setminus \{ \emptyset \} .
$$

The second, called the minimal building set, consists of all nonempty connected flats, i.e.,

$$
\mathfrak { G } _ { \mathrm { m i n } } : = \{ F \in \mathcal { L } ( \mathsf { M } ) \setminus \{ \varnothing \} : \mathsf { M } | _ { F } \mathrm { ~ i s ~ c o n n e c t e d } \} .
$$

Recall here that a polymatroid $\mathsf { M }$ is connected if the lattice $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ cannot be nontrivially decomposed as a Cartesian product.

For a flat $F$ of a polymatroid $\mathsf { M } = ( E , \mathrm { r k } )$ , the restriction $\mathsf { M } | _ { F }$ is the polymatroid $\bigl ( F , \operatorname { r k } | _ { F } \bigr )$ , and the contraction $\mathsf { M } / F$ is the polymatroid $( E \setminus F , \mathrm { r k } ^ { \prime } )$ where $\operatorname { r k } ^ { \prime } ( A ) : = \operatorname { r k } ( A \cup$ $F ) - \operatorname { r k } ( F )$ . The lattices of flats of $\mathsf { M } | _ { F }$ and $\mathsf { M } / F$ are isomorphic to the intervals $[ \sigma , F ]$ and $[ F , E ]$ of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ , respectively. These operations extend to building sets, as follows.

Definition 2.3 Given a flat $F$ of $\mathsf { M }$ and a building set $\mathcal { G }$ of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ , we define the sets

$$
\begin{array} { r } { \pmb { \mathcal { G } } | _ { F } : = \{ G \in \pmb { \mathcal { G } } : G \subseteq F \} \subseteq \pmb { \mathcal { L } } ( \mathbf { M } | _ { F } ) } \end{array}
$$

and

$$
\begin{array} { r } { \ L \mathcal { G } / F : = \{ G \lor F : G \in \mathcal { G } \mathrm { ~ a n d ~ } G \mathcal { G } F \} \subseteq \mathcal { L } ( \mathsf { M } / F ) } \end{array}
$$

where ∨ stands for the join operation of a lattice.

It can be proved that $\mathcal { G } | _ { F }$ is a building set on $\mathcal { L } ( \mathsf { M } | _ { F } )$ and $\mathcal { G } / F$ is a building set on $\mathcal { L } ( \boldsymbol { \mathsf { M } } / F )$ . Note that $\mathcal { G } | _ { F } = \mathcal { G } _ { \leq F }$ ; we will use the symbol $\mathcal { G } | _ { F }$ when we want to emphasize that we are considering a building set on the restricted polymatroid.

Remark 2.4 If $\mathcal { G } = \mathcal { G } _ { \operatorname* { m a x } }$ is the maximal building set for $\mathsf { M }$ , then $\mathcal { G } | _ { F }$ and $\mathcal { G } / F$ are maximal building sets of the restriction and contraction respectively. We caution however that when $\mathcal { G } = \mathcal { G } _ { \operatorname* { m i n } }$ is the minimal building set, $\mathcal { G } / F$ may not be the minimal building set in the contraction, since contracting a flat may not preserve connectivity.

2.2. Nested sets and Chow rings. The choice of a building set $\mathcal { G }$ on the lattice of flats $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ of a polymatroid M induces a special class of sets called $\mathcal { G }$ -nested sets. They are defined as follows.

Definition 2.5 A subset $S \subseteq { \mathcal { G } }$ is said to be $\mathcal { G }$ -nested if, for each subset of incomparable flats $\{ F _ { 1 } , \ldots , F _ { m } \} \subseteq S$ of cardinality at least 2, the join $F _ { 1 } \lor \cdots \lor F _ { m }$ does not lie in $\mathcal { G }$ .

Clearly, if $S$ is $\mathcal { G }$ -nested, then any subset of $S$ is $\mathcal { G }$ -nested. In particular, the family of all $\mathcal { G }$ -nested sets forms a simplicial complex that we will denote $\mathcal { N } ( \mathsf { M } , \mathsf { \mathcal { G } } )$ and call the nested set complex associated to $\mathcal { G }$ . When $\mathcal { G } = \mathcal { G } _ { \operatorname* { m a x } }$ , the nested set complex coincides with the order complex of ${ \mathcal { L } } ( \mathsf { M } ) \setminus \{ \emptyset \}$ .

The central objects of study in this paper are the Chow rings arising from a polymatroid and a building set. These rings (in a slightly more general set up) were introduced by Feichtner and Yuzvinsky in [FY04].

Definition 2.6 Let $\mathcal { G }$ be a building set on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . We define the Chow ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ as the graded quotient algebra

$$
D ( { \mathcal { L } } ( { \mathsf { M } } ) , { \mathcal { G } } ) = \mathbb { Q } [ x _ { G } : G \in { \mathcal { G } } ] { \bigg / } I ,
$$

where $I$ is the homogeneous ideal generated by the relations

$$
\prod _ { i = 1 } ^ { m } x _ { G _ { i } } \qquad \mathrm { f o r } ~ \{ G _ { 1 } , \dots , G _ { m } \} ~ \mathrm { n o t } ~ { \mathcal { G } } _ { \mathrm { - n e s t e d , } }
$$

and

$$
\sum _ { G \geq F } x _ { G } \qquad { \mathrm { f o r ~ } } F { \mathrm { ~ a n ~ a t o m ~ o f ~ } } { \mathcal { L } } ( \mathsf { M } ) .
$$

The rings $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ are graded and Artinian, i.e., they are finite dimensional as $\mathbb { Q }$ -vector spaces, and under the decomposition $D ( \mathsf { M } , \mathsf { \mathcal { G } } ) = \bigoplus _ { i = 0 } ^ { \operatorname { r k } ( E ) - 1 } D ^ { i } ( \mathsf { M } , \mathsf { \mathcal { G } } )$ Lrk(E)−1i=0 Di(M, G) given by the grading, we have that $D ^ { \iota } ( { \mathsf { M } } , { \mathcal { G } } ) = 0$ for all $i \geq \operatorname { r k } ( E )$ . Recall our notation for the Hilbert series:

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { G } } } ( x ) : = \sum _ { i = 0 } ^ { \operatorname { r k } ( E ) - 1 } \dim \left( D ^ { i } ( \mathsf { M } , \mathsf { \mathcal { G } } ) \right) x ^ { i } .
$$

One of the main results by Feichtner and Yuzvinsky, which follows from the construction of a Gr¨obner basis for the ideal $I$ in Definition 2.6, is that the rings $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ possess an additive basis given by a specific list of monomials. Since this is useful for computing the Hilbert series, we restate it here.

Theorem 2.7 ([FY04, Corollary 1] and [PP23, Corollary 2.8]) For every polymatroid M and building set $\mathcal { G }$ , the following monomials constitute a basis of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ as a $\mathbb { Q }$ -vector space:

$$
x _ { F _ { 1 } } ^ { e _ { 1 } } x _ { F _ { 2 } } ^ { e _ { 2 } } \cdot \cdot \cdot x _ { F _ { m } } ^ { e _ { m } } ,
$$

where $S : = \{ F _ { 1 } , \ldots , F _ { m } \}$ is $\mathcal { G }$ -nested, and

$$
0 \leq e _ { i } < \mathrm { r k } ( F _ { i } ) - \mathrm { r k } ( \operatorname* { s u p } ( S , F _ { i } ) )
$$

for each $1 \leq i \leq m$ , where $\operatorname* { s u p } ( S , F ) : = \bigvee _ { G \in S \atop G < F } G$

The monomials appearing in the preceding statement will be customarily called the “FY-monomials” associated to $\mathsf { M }$ and $\mathcal { G }$ . Notice that when $\mathcal { G } = \mathcal { G } _ { \operatorname* { m a x } }$ , a nested set is just a flag of flats $S = \{ F _ { 1 } \subsetneq \cdots \subsetneq F _ { m } \}$ , and the rank condition in the theorem reads $0 \leq e _ { i } \leq \mathrm { r k } ( F _ { i } ) - \mathrm { r k } ( F _ { i - 1 } ) - 1$ for all $1 \leq i \leq m$ .

The main goal of this section is to extend [FMSV24, Theorem 1.4] to general building sets. To this end, we will work in the incidence algebra of the lattice of flats of a polymatroid. Specifically, if $\mathcal { L } ( \boldsymbol { M } )$ is the lattice of flats of a polymatroid, the incidence algebra $\mathcal { I }$ of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ is the free $\mathbb { Z } [ x ]$ -module spanned by all closed intervals of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . The product of two elements $a , b \in { \mathcal { I } }$ is given by the convolution

$$
( a \cdot b ) _ { F G } : = \sum _ { H \in [ F , G ] } a _ { F H } \cdot b _ { H G } ,
$$

for every choice of flats $F \subseteq G$ , where $a F G$ denotes the polynomial that $a$ associates to the closed interval $[ F , G ]$ . The two-sided identity in this algebra is the element $\delta \in \mathcal { I }$ defined by $\delta _ { F G } : = 1$ for $F = G$ and 0 otherwise. Another very useful object that we will use repeatedly is the so-called zeta function $\zeta \in \mathcal { I }$ , which is defined by $\zeta _ { F G } : = 1$ for every $F \subseteq G$ . We will assume that the reader is familiar with the basics of incidence algebras, and we refer to [Sta12, Chapter 3] for undefined terminology.

3.1. $\mathcal { G }$ -reduced characteristic polynomials. We now introduce a generalization of the reduced characteristic polynomial, which takes into account general building sets.

Definition 3.1 Let $\mathsf { M }$ be a polymatroid on $E$ , and let $\mathcal { G }$ be a building set on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . The $\mathcal { G }$ -reduced characteristic polynomial of $\mathsf { M }$ is defined as

$$
\overline { { \chi } } _ { \mathsf { M } } ^ { \sharp } ( x ) : = \left\{ \begin{array} { l l } { - 1 } & { \mathrm { i f ~ \mathsf { M } ~ i s ~ t h e ~ e m p t y ~ p o l y m a t r o i d , ~ a n d } } \\ { - \frac { \chi _ { \mathsf { M } } ( x ) } { ( 1 - x ) ^ { | f ( \sharp ) | } } } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

If $\mathsf { M }$ is nonempty and $\mathcal { G } = \mathcal { G } _ { \operatorname* { m a x } }$ , then $| f ( \mathcal { G } ) | = | \{ E \} | = 1$ , so the $\mathcal { G } _ { \mathrm { m a x } }$ -reduced characteristic polynomial agrees with the usual reduced characteristic polynomial $\overline { { \chi } } _ { \mathsf { M } } ( x ) : =$ $\chi _ { \mathsf { M } } ( x ) / ( x - 1 )$ . Since a building set $\mathcal { G }$ on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ induces building sets on each interval $[ F , G ] \subseteq { \mathcal { L } } ( \mathsf { M } )$ , we can define the following key object in the incidence algebra of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ .

Definition 3.2 Let $\mathcal { G }$ be a building set on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . The $\mathcal { G }$ -reduced characteristic function on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ is the incidence algebra element $\overline { { \chi } } ^ { \mathcal { G } } \in \mathcal { I }$ defined by

$$
\overline { { \chi } } _ { F G } ^ { \mathcal { G } } : = \overline { { \chi } } _ { \sf M | _ { G } / F } ^ { \mathcal { G } | _ { G } / F } ( x )
$$

for every pair of flats $F \subsetneq G$ of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ , and $\overline { { \chi } } _ { F G } ^ { \mathcal { G } } : = - 1$ whenever $F = G$ .

The unusual choice for how to define the $\mathcal { G }$ -reduced characteristic polynomial of a singleton interval is important, and it is motivated by a similar choice made in [FMSV24] of defining the usual reduced characteristic polynomial of an empty matroid as $- 1$ . Before proving our main results, we need a further ingredient.

Definition 3.3 Let $\mathsf { M }$ be a polymatroid on $E$ . The $\alpha$ -polynomial associated to a building set $\mathcal { G }$ on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ is defined as follows:

$$
\alpha _ { \mathsf { M } } ^ { \mathsf { g } } ( x ) : = ( - 1 ) ^ { | f ( \mathsf { g } ) | - 1 } \prod _ { F \in f ( \mathsf { \mathcal { G } } ) } ( x + x ^ { 2 } + \cdot \cdot \cdot + x ^ { \mathrm { r k } ( F ) - 1 } ) .
$$

The $\alpha$ -function is the element $\alpha ^ { \mathcal { G } }$ in the incidence algebra $\mathcal { I }$ defined by

$$
\alpha _ { F G } ^ { \mathcal { G } } : = \alpha _ { \mathsf { M } | _ { G } / F } ^ { \mathcal { G } | _ { G } / F } ( x )
$$

for every pair of flats $F \subsetneq G$ , and $\alpha _ { F G } : = - 1$ whenever $F = G$ .

The following lemma will play an important role in the proof of the main theorems in this section.

Lemma 3.4 We have the following identity in the incidence algebra:

$$
{ \boldsymbol { \zeta } } \cdot { \overline { { \chi } } } ^ { \mathcal { G } } = \alpha ^ { \mathcal { G } } .
$$

Proof. It follows from the definitions that for every pair of flats $F \subseteq G$ of $\mathsf { M }$ , we have $f ( \mathscr { G } | _ { G } / F ) = f ( \mathscr { G } | _ { G } ) \setminus f ( \mathscr { G } | _ { F } )$ . Let us set $f ( \mathcal { G } | _ { G } ) = \{ G _ { 1 } , G _ { 2 } , \dots , G _ { k } \}$ .

$$
\begin{array} { r l } { \langle \boldsymbol { \mathcal { X } } : \boldsymbol { \mathcal { X } } ^ { 0 } \rangle _ { \mathrm { F L G } ^ { \ell - 1 } } - } & { \displaystyle \sum _ { F \leq i \leq L } \frac { 1 } { \Gamma ( \boldsymbol { \mathcal { X } } ^ { 0 } ) \leq \Gamma ^ { 1 / 2 } } \frac { X \mu ( \boldsymbol { \mathcal { O } } ^ { ( 1 ) } ) } { \Gamma ^ { 1 / 2 } } } \\ { = } & { - \displaystyle \sum _ { F \leq i \leq R } \displaystyle \sum _ { \boldsymbol { \mathcal { X } } ^ { 0 } \geq \boldsymbol { \mathcal { X } } ^ { 0 } \geq \boldsymbol { \mathcal { X } } ^ { 0 } } \frac { 1 } { \Gamma ^ { 1 / 2 } - 1 } \frac { X \mu ( \boldsymbol { \mathcal { O } } ^ { ( 1 ) } ) } { 1 - \boldsymbol { \mathcal { X } } ^ { 0 } } } \\ & { - \displaystyle \sum _ { F \leq i \leq R } \prod _ { \boldsymbol { \mathcal { X } } ^ { 0 } \geq \boldsymbol { \mathcal { X } } ^ { 0 } \geq \boldsymbol { \mathcal { X } } ^ { 0 } } \Bigg ( 1 - \overline { { X } } _ { \boldsymbol { \mathcal { O } } ^ { 0 } } ( \boldsymbol { \mathcal { X } } ^ { 0 } ) } \\ & { - \frac { 1 } { \Gamma ^ { 1 / 2 } } \mathrm { R e } \mathrm { i } \overline { { ( \boldsymbol { \mathcal { X } } ^ { 0 } + \boldsymbol { \mathcal { X } } ^ { 0 } ) } } } \\ &  - \displaystyle \lbrace - 1 ) ^  \mathrm { f } / \mathrm { f } / \mathrm { f } / \mathrm { f } / \mathrm { f } \mathrm { f } / \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \mathrm { f } \ \end{array}
$$

The sum in the second line is over all choices of flats $( F _ { 1 } , \dots , F _ { k } ) \in [ F , G _ { 1 } ] \times \dots \times [ F , G _ { k } ]$ , and the factor inside is set to be 1 when $F \nsubseteq G _ { i }$ for some $i$ ; this explains the appearance of exactly $\left| f ( \mathcal { G } | _ { G } / F ) \right|$ minus signs in the fourth line. Meanwhile, in the last line we used [FMSV24, Lemma 2.5]. 

# 3.2. Recursions for Hilbert series of Chow rings.

Definition 3.5 Let $\mathcal { G }$ be a building set on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . The $\mathcal { G }$ -Chow polynomial of a nonempty polymatroid M is defined as the Hilbert series $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ . The $\mathcal { G }$ -Chow function is the element $\mathrm { H } ^ { \mathcal { G } }$ in the incidence algebra $\mathcal { I }$ of $\mathcal { L } ( \boldsymbol { M } )$ defined by

$$
\begin{array} { r } { \mathrm { H } _ { F G } ^ { \mathcal { G } } : = \mathrm { H } _ { \mathsf { M } | G / F } ^ { \mathcal { G } | _ { G } / F } ( x ) , } \end{array}
$$

for every pair of flats $F \subsetneq G$ , and $\mathrm { H } _ { F G } ^ { \mathcal { G } } : = 1$ whenever $F = G$

The following is the crucial result enabling the computation of Hilbert series of Chow rings using arbitrary building sets.

Theorem 3.6 Let $\mathcal { G }$ be a building set on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . The $\mathcal { G }$ -Chow function and the $\mathcal { G }$ -reduced characteristic function are inverses of each other in $\mathcal { I }$ up to a sign. That is,

$$
\begin{array} { r } { \mathrm { H } ^ { \mathcal { G } } = - \left( \overline { { \chi } } ^ { \mathcal { G } } \right) ^ { - 1 } . } \end{array}
$$

Proof. First, let us show that $\alpha ^ { \mathcal { G } } \cdot \mathrm { H } ^ { \mathcal { G } } = - \zeta$ . For the sake of simplicity, and without loss of generality, we will prove that $( \alpha ^ { \mathcal { G } } \cdot \mathrm { H } ^ { \mathcal { G } } ) _ { \mathcal { O } E } = - \zeta _ { \mathcal { O } E }$ . The case of more general intervals follows from this one.

$h ( F ) \in [ 1 , \operatorname { r k } ( F ) - 1 ]$ For each $G \in { \mathcal { L } } ( { \mathsf { M } } )$ . The function  total degree of , consider the set $\alpha _ { \scriptscriptstyle O G } ^ { \mathcal { G } | _ { G } }$ is (up to monomial of functions $h \colon f ( \mathcal { G } | _ { G } ) \to \mathbb { N }$ rating function of . The left hand s such that $A ( G )$   
with respect to the the QF ∈f(G|G) xh(FF   
the identity we claim is

$$
\sum _ { G \in \mathcal { L } ( \mathsf { M } ) } \alpha _ { \emptyset G } ^ { \mathcal { G } } \mathrm { H } _ { G E } ^ { \mathcal { G } } ,
$$

and it counts (without considering the signs) the possible choices of an element of $A ( G )$ and of an FY-monomial in $[ G , E ]$ . Notice that, for each $G$ , there is a bijection $\psi$ between $\mathcal { G } / G$ -nested sets and $\mathcal { G }$ -nested sets containing $f ( \mathscr { G } | _ { G } )$ . Let $\mathrm { F Y } ( \mathsf { M } , \mathcal { G } )$ be the set of FYmonomials associated to $\mathsf { M }$ and $\mathcal { G }$ , and define an injection

$$
\varphi _ { G } \colon A ( G ) \times \mathrm { F Y } ( \mathsf { M } / G , \mathcal { G } / G )  \mathrm { F Y } ( \mathsf { M } , \mathcal { G } )
$$

by

$$
\varphi _ { G } \left( h , \prod _ { F \in { \cal N } } x _ { F } ^ { k ( F ) } \right) = \prod _ { F \in f ( \mathcal { G } | _ { G } ) } x _ { F } ^ { h ( F ) } \prod _ { F \in { \cal N } } x _ { \psi ( F ) } ^ { k ( F ) } .
$$

The range of $\varphi _ { G }$ is the set of all FY-monomials associated to a nested set $N$ such that $f ( \mathcal G | _ { G } ) \subseteq \operatorname* { m i n } ( N )$ . The sets $\operatorname { I m } \varphi _ { G }$ cover $\mathrm { F Y } ( \mathsf { M } , \mathcal { G } )$ and an FY-monomial $m ( N , k ) =$ QF ∈N xk(F is contained in $2 ^ { | \mathrm { m i n } ( N ) | }$ sets corresponding to the element ∨ $T$ for any $T \subseteq$ $\operatorname* { m i n } ( N )$ . Therefore,

$$
( \alpha \cdot \mathrm { H } ) _ { \mathcal { Q } E } ^ { \mathcal { G } } = \sum _ { m ( N , K ) \in \mathrm { F Y } ( \mathrm { M } , \mathcal { G } ) } \left( \sum _ { T \subseteq \operatorname* { m i n } ( N ) } ( - 1 ) ^ { | T | - 1 } \right) x ^ { \mathrm { d e g } ( m ( N , k ) ) } = - 1 .
$$

This completes the proof.

3.3. Explicit formulas for the Hilbert series. Using Theorem 3.6, we can deduce the following recursions for computing Hilbert series of Chow rings of polymatroids with arbitrary building sets, which is one of the main results of the present paper.

Theorem 1.1 For every nonempty polymatroid M and every building set $\mathcal { G }$ , the following two recursions hold:

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { E } } } ( x ) = \sum _ { \substack { \emptyset \neq F \in \mathbf { \mathcal { L } } ( \mathsf { M } ) } } \frac { - \chi _ { \mathsf { M } | _ { F } } ( x ) } { ( 1 - x ) ^ { | f ( \mathfrak { G } | _ { F } ) | } } \cdot \mathrm { H } _ { \mathsf { M } / F } ^ { \mathsf { \mathcal { G } } / F } ( x ) ,
$$

and

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { G } } } ( x ) = \sum _ { E \neq F \in { \mathcal { L } } ( \mathsf { M } ) } \mathrm { H } _ { \mathsf { M } | _ { F } } ^ { \mathsf { { g } } | _ { F } } ( x ) \cdot \frac { - \chi _ { \mathsf { M } / F } ( x ) } { ( 1 - x ) ^ { | f ( \mathsf { \mathcal { G } } / F ) | } } .
$$

Proof. Because of how we defined the element $\mathbf { H } ^ { \mathcal { G } } \in \mathcal { I }$ , the Hilbert series of the ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ equals $\mathrm { H } _ { \mathcal { O E } } ^ { \mathcal { G } }$ . By applying Theorem 3.6, we have that $\overline { { \chi } } ^ { \mathcal { G } } \cdot \mathrm { H } ^ { \mathcal { G } } = - \delta$ . In particular, since $E \neq \emptyset$ , we have that

$$
\begin{array} { r } { \left( \overline { { \chi } } ^ { \mathcal { G } } \cdot \mathrm { H } ^ { \mathcal { G } } \right) _ { \varnothing E } = 0 . } \end{array}
$$

Expanding, we obtain that

$$
\sum _ { F \in \mathcal { L } ( \mathsf { M } ) } \overline { { \chi } } _ { \mathcal { O } F } ^ { \mathcal { G } } \mathrm { H } _ { F E } ^ { \mathcal { G } } = 0 ,
$$

and the identity follows from separating the summand corresponding to $F = \varnothing$ .

The proof of the other identity is completely analogous, but using instead that $\mathbf { H } ^ { \mathcal { G } } \cdot \overline { { \chi } } ^ { \mathcal { G } } =$ $- \delta$ and isolating the summand corresponding to $F = E$ . 

![](images/c6e8bd1d8158212a29646e26844cfebd2045a97290d93f5cd6566092ae1ec189.jpg)  
Figure 2. The lattice of flats of a polymatroid M

Example 3.7 Consider the loopless polymatroid $\mathsf { M }$ on $E = \{ a , b , c \}$ described in [PP23, Section 7]. Its lattice of flats is depicted in Figure 2. Consider the building set ${ \mathcal { G } } =$ $\{ a , b , c , a b c \}$ . One can enumerate the FY-monomials to see that

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = 1 + 4 x + 5 x ^ { 2 } + 4 x ^ { 3 } + x ^ { 4 } .
$$

$\mathrm { H } _ { \mathsf { M } / F } ^ { \mathcal { G } / F } ( x )$ ow how to compute this polynomial using Theorem 1.1. We begin by computing in descending order of rank. Since for polymatroids of rank at most 1 the Hilbert series of the Chow ring is identically 1, we have that $\mathrm { H } _ { \mathsf { M } / a b c } ^ { \mathcal { G } / a b c } ( x ) = \mathrm { H } _ { \mathsf { M } / a b } ^ { \mathcal { G } / a b } ( x ) = \mathrm { H } _ { \mathsf { M } / c } ^ { \mathcal { G } / c } ( x ) = 1$ We can now compute $\mathrm { H } _ { \mathsf { M } / a } ^ { \mathcal { G } / a } ( x )$ (and $\mathrm { H } _ { \mathsf { M } / b } ^ { \mathcal { G } / b } ( x )$ in an identical fashion) as

$$
\begin{array} { l } { { \displaystyle { \mathrm { H } } _ { \mathrm { M } / a } ^ { \mathcal { G } / a } ( x ) = - \frac { \chi _ { \mathsf { M } | _ { a b } / a } ( x ) } { ( 1 - x ) ^ { | f ( \mathcal { G } | _ { a b } / a ) | } } { \mathrm { H } } _ { \mathrm { M } / a b } ^ { \mathcal { G } / a b } ( x ) - \frac { \chi _ { \mathsf { M } / a } ( x ) } { ( 1 - x ) ^ { | f ( \mathcal { G } / a ) | } } H _ { \mathsf { M } / a b c } ^ { \mathcal { G } / a b c } ( x ) } }  \\ { { \displaystyle \qquad = - \frac { x ^ { 2 } - 1 } { 1 - x } - \frac { x ^ { 3 } - x } { 1 - x } } } \\ { { \displaystyle \qquad = 1 + 2 x + x ^ { 2 } . } } \end{array}
$$

Lastly, we observe that $| f ( \mathcal { G } | _ { a b } ) | = 2$ , so

$$
{ \frac { \chi _ { \sf M | _ { a b } } ( x ) } { ( 1 - x ) ^ { | f ( \mathcal { G } | _ { a b } ) | } } } = { \frac { ( x ^ { 2 } - 1 ) ^ { 2 } } { ( 1 - x ) ^ { 2 } } } = ( x + 1 ) ^ { 2 } .
$$

Therefore,

$$
\begin{array} { l } { { \displaystyle { \mathrm { H } } _ { \mathrm { M } } ^ { \mathfrak { g } } ( x ) = - \frac { \chi _ { \mathrm { M } \vert _ { a } } ( x ) } { 1 - x } { \mathrm { H } } _ { \mathrm { M } / a } ^ { \mathfrak { g } / { a } } ( x ) - \frac { \chi _ { \mathrm { M } \vert _ { b } } ( x ) } { 1 - x } { \mathrm { H } } _ { \mathrm { M } / b } ^ { \mathfrak { g } / { b } } ( x ) - \frac { \chi _ { \mathrm { M } \vert _ { c } } ( x ) } { 1 - x } { \mathrm { H } } _ { \mathrm { M } / c } ^ { \mathfrak { g } / { c } } ( x ) } } \\ { { \displaystyle ~ - \frac { \chi _ { \mathrm { M } \vert _ { a b } } ( x ) } { ( 1 - x ) ^ { 2 } } { \mathrm { H } } _ { \mathrm { M } / a b } ^ { \mathfrak { g } / { a b } } ( x ) - \frac { \chi _ { \mathrm { M } } ( x ) } { 1 - x } } } \\ { { \displaystyle ~ = ( x + 1 ) ( x + 1 ) ^ { 2 } + ( x + 1 ) ( x + 1 ) ^ { 2 } + ( x ^ { 3 } + x ^ { 2 } + x + 1 ) \cdot 1 } } \\ { { \displaystyle ~ - ( x + 1 ) ^ { 2 } \cdot 1 + ( x ^ { 4 } + x ^ { 3 } - x ^ { 2 } - x - 1 ) } } \\ { { \displaystyle ~ = 1 + 4 x + 5 x ^ { 2 } + 4 x ^ { 3 } + x ^ { 4 } } . } \end{array}
$$

We now rewrite Theorem 3.6 to obtain a non-recursive formula for $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ as a sum over chains of flats.

Corollary 1.2 The Hilbert series of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ equals

$$
\mathrm { H } _ { \mathrm { M } } ^ { \mathcal { G } } ( x ) = \sum _ { m \geq 1 } \sum _ { \mathcal { O } = F _ { 0 } \subset \cdots \subsetneq F _ { m } = E } ( - 1 ) ^ { m } \prod _ { i = 1 } ^ { m } \frac { \chi _ { \mathrm { M } | _ { F _ { i } } / F _ { i - 1 } } ( x ) } { ( 1 - x ) ^ { | f ( \mathcal { G } | _ { F _ { i } } / F _ { i - 1 } ) | } } .
$$

Proof. Combining the identity in Theorem 3.6 with a general formula describing inverses in the incidence algebra as sums over chains (see [ER95, Lemma 5.3]) gives the result. $\boxed { \begin{array} { r l } \end{array} }$

Example 3.8 Consider ${ \mathsf { M } } = { \mathsf { K } } _ { 4 }$ , the graphic matroid associated to the complete graph on 4 vertices. As we explain at the beginning of Section 4, flats of this matroid are in natural bijection with partitions of $\{ 1 , 2 , 3 , 4 \}$ . These are the possible chains in $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ , where we denote flats by partitions of [4]:

• one flag of type $1 | 2 | 3 | 4 \subsetneq 1 2 3 4$ , • 6 flags of type $1 | 2 | 3 | 4 \subsetneq i j | k | l \subsetneq 1 2 3 4$ , • 4 flags of type $1 | 2 | 3 | 4 \subsetneq i j k | l \subsetneq 1 2 3 4$ , • 3 flags of type $1 | 2 | 3 | 4 \subsetneq i j | k l \subsetneq 1 2 3 4$ , • 12 flags of type $1 | 2 | 3 | 4 \subsetneq i j | k | l \subsetneq i j k | l \subsetneq 1 2 3 4$ , $\bullet$ 6 flags of type $1 | 2 | 3 | 4 \subsetneq i j | k | l \subsetneq i j | k l \subsetneq 1 2 3 4$ .

Let us write $\chi _ { m } ( x ) = ( x - 1 ) ( x - 2 ) \cdot \cdot \cdot ( x - m + 1 )$ for the characteristic polynomial of $\mathsf { K } _ { m }$ . The formula in the preceding corollary states that

$$
\begin{array} { r l } & { \mathrm { H } _ { \mathrm { K } _ { 4 } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x ) = \overline { { \chi } } _ { 4 } + 6 \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 3 } + 4 \overline { { \chi } } _ { 3 } \overline { { \chi } } _ { 2 } - 3 \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 2 } + 1 2 \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 2 } + 6 \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 2 } \overline { { \chi } } _ { 2 } } \\ & { \qquad = ( x - 2 ) ( x - 3 ) + 1 0 ( x - 2 ) + 1 5 } \\ & { \qquad = x ^ { 2 } + 5 x + 1 . } \end{array}
$$

3.4. A more efficient non-recursive formula. Let $S$ be a $\mathcal { G }$ -nested set. We recall from the notation in the statement of Theorem 2.7 that

$$
\operatorname* { s u p } ( S , F ) : = \bigvee _ { F _ { j } \in S \atop F _ { j } < F } F _ { j } .
$$

Following the terminology in [Cor25a, Definition 2.12], we say that a $\mathcal { G }$ -nested set $S$ is spanning if the join of all the elements in $S$ is the ground set $E$ or, equivalently, if $S$ contains all the maximal elements of $\mathcal { G }$ . We have the following formula for computing $\mathrm { H } _ { \mathsf { M } } ^ { \lessgtr } ( x )$ in terms of spanning $\mathcal { G }$ -nested sets.

Theorem 1.3 The Hilbert series of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ equals

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = \sum _ { S } \prod _ { F \in S } \overline { { \chi } } _ { \mathsf { M } | _ { F } / \operatorname* { s u p } ( S , F ) } ( x ) ,
$$

where the sum runs over all spanning $\mathcal { G }$ -nested sets.

Proof. We show this by induction on the rank of the polymatroid M. We analyze two cases according to whether $E$ belongs to $\mathcal { G }$ or not.

If $E \in { \mathcal { G } }$ , then, by the induction hypothesis applied to the recursion (4), we have

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { G } } } ( x ) = \sum _ { E \neq F \in \mathcal { L } ( \mathsf { M } ) } \left( \sum _ { \begin{array} { l } { S ^ { \prime } \mathrm { s p a n n i n g ~ } G \in S ^ { \prime } } \\ { \mathcal { G } _ { \mid F ^ { - \mathrm { n e s t e d } } } } \end{array} } \prod _ { G \in S ^ { \prime } } \overline { { \chi } } _ { \mathsf { M } \mid G } / \operatorname* { s u p } ( S ^ { \prime } , G ) \right) \overline { { \chi } } _ { \mathsf { M } / F } ^ { \mathcal { G } / F } ( x ) .
$$

Since $E \in { \mathcal { G } }$ , we have that $\overline { { \chi } } _ { \sf M / F } ^ { \mathcal { G } / F } ( x ) = \overline { { \chi } } _ { \sf M / F } ( x )$ . Notice that for each spanning $\mathcal { G } | _ { F }$ -nested set $S ^ { \prime }$ appearing in the sum between parentheses, the set $S = S ^ { \prime } \cup \{ E \}$ is $\mathcal { G }$ -nested and $F = \operatorname* { s u p } ( S , E )$ . Therefore, we can write $\overline { { \chi } } _ { \mathsf { M } / F } ( x ) = \overline { { \chi } } _ { \mathsf { M } | _ { E } / \operatorname { s u p } ( S , E ) } ( x )$ and group it with the factors inside the sum, thus finishing the induction step in this case.

If, instead, $E \notin { \mathcal { G } }$ let $f ( { \mathcal { G } } ) = \{ F _ { 1 } , \ldots , F _ { k } \}$ be the $\mathcal { G }$ -factors. There is a correspondence between $\mathcal { G }$ -nested sets and the disjoint union of $\mathcal { G } | _ { F _ { i } }$ -nested sets for $i = 1 , \dots k$ ; moreover, this correspondence extends to FY-monomials. Hence, the identities

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = \prod _ { j = 1 } ^ { k } \mathrm { H } _ { \mathsf { M } | _ { F _ { j } } } ^ { \mathcal { G } | _ { F _ { j } } } ( x )
$$

and

$$
\sum _ { \substack { S \mathrm { \scriptsize ~ s p a n n i n g } } } \prod _ { \substack { F \in S } } \overline { { \chi } } _ { \mathsf { M } | _ { F } / \operatorname* { s u p } ( S , F ) } = \prod _ { j = 1 } ^ { k } \sum _ { \substack { S \mathrm { \scriptsize ~ s p a n n i n g } } } \prod _ { \substack { F \in S } } \overline { { \chi } } _ { \mathsf { M } | _ { F } / \operatorname* { s u p } ( S , F ) }
$$

hold and allow us to conclude by using the induction hypothesis.

3.5. The geometry of the formulas. We explain the geometry behind our formulas for $\mathrm { H } _ { \mathsf { M } } ^ { \lessgtr } ( x )$ . For simplicity, let us explain the recursion (4) under the assumption that $\mathsf { M }$ is a matroid and that $\mathcal { G }$ contains the ground set $E$ . This geometric picture will also be helpful in constructing examples in Section 6.

Suppose the matroid $\mathsf { M }$ has a realization by a linear subspace $L \subseteq \mathbb { C } ^ { E }$ . That is, the rank function of $\mathsf { M }$ is given by

$\operatorname { r k } ( S ) = \dim ( \operatorname { t h e ~ i m a g e ~ o f ~ } L \operatorname { ~ u n d e r ~ t h e ~ p r o j e c t i o n ~ } \mathbb { C } ^ { E } \to \mathbb { C } ^ { S } ) .$

We assume that $L$ is not contained in a coordinate hyperplane, i.e. that $\mathsf { M }$ is loopless. Denote by $\mathbb { P } \overset { \circ } { L } : = \mathbb { P } L \cap \mathbb { P } ( ( \mathbb { C } ^ { * } ) ^ { E } )$ the projective hyperplane arrangement complement. For $F$ a flat of $\mathsf { M }$ , let $L _ { F } : = L \cap \{ x _ { i } = 0 : i \in F \}$ and $L ^ { F } : = L / L _ { F }$ . We consider $L _ { F }$ and $L ^ { F }$ as subspaces of $\mathbb { C } ^ { E \backslash F }$ and $\mathbb { C } ^ { F }$ respectively, so that they define arrangement complements $\mathbb { P } \overset { \circ } { L } _ { F }$ and $\mathbb { P } \mathring { L } ^ { F }$ , with corresponding matroids $\mathsf { M } / F$ and $\mathsf { M } | _ { F }$ , respectively. Note that $\| ^ { \flat } L _ { E } = \emptyset$ and $\mathbb { P } L ^ { E } = \mathbb { P } L$ .

Given a building set $\mathcal { G }$ containing $E$ , let $\{ G _ { 1 } , \dots , G _ { m } \}$ be a total order on $\mathcal { G }$ that refines the order on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ . The wonderful variety $\underline { W } _ { L } ^ { \lessgtr }$ introduced in [DCP95] is the sequential blow-up of (the strict transforms of) the $\mathbb { P } L _ { G _ { i } }$ , that is,

$$
\underline { { W } } _ { L } ^ { \mathcal { G } } : = \operatorname { B l } _ { \widetilde { \mathbb { P } } L _ { G _ { 1 } } } \big ( \cdot \cdot \cdot ( \operatorname { B l } _ { \widetilde { \mathbb { P } } L _ { G _ { m - 1 } } } ( \operatorname { B l } _ { \mathbb { P } L _ { G _ { m } } } \mathbb { P } L ) ) \cdot \cdot \cdot \big ) .
$$

The cohomology ring of $\underline { W } _ { L } ^ { \lessgtr }$ is isomorphic to $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$

Let $P _ { X } ( q )$ denote the virtual Poincar´e polynomial of a variety $X$ over $\mathbb { C }$ , characterized by the following two properties (see, for instance, [Ful93, Section 4.5]): (i) If $X$ is complete and smooth, then $\begin{array} { r } { P _ { X } ( q ) = \sum _ { i \geq 0 } \dim { H ^ { i } } ( X , \mathbb { Q } ) q ^ { i } } \end{array}$ ; and (ii) if $\begin{array} { r } { X = \bigcup _ { j } Y _ { j } } \end{array}$ is a decomposition of $X$ into finitely many locally closed subsets $Y _ { j }$ , then $\begin{array} { r } { P _ { X } = \sum _ { j } P _ { Y _ { j } } } \end{array}$ . Note that $P _ { \underline { { W } } _ { L } ^ { \mathcal { G } } } ( q ) \ = \ \mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( q ^ { 2 } )$ . Since $E \in { \mathcal { G } }$ , the recursion (4) states

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathsf { \mathcal { G } } } ( x ) = \sum _ { F \neq E } \mathrm { H } _ { { \mathsf { M } } | _ { F } } ^ { \mathsf { \mathcal { G } } | _ { F } } ( x ) \cdot \overline { { \chi } } _ { \mathsf { M } / F } ( x ) ,
$$

which now follows from the following observations:

• The projection map $\pi \colon \underline { W } _ { L } ^ { \mathcal G } \to \mathbb P L$ is a composition of blow-down maps, under   
which the strict transform of $\mathbb { P } L _ { F }$ is × W L [ DCP95, Theorem 4.3], so that   
π−1(P˚LF ) ≃ W LF × .   
• We have a decomposition $\begin{array} { r } { \mathbb { P } L = \bigcup _ { F \neq E } \mathbb { P } \overset { \circ } { L } _ { F } } \end{array}$ into locally closed subvarieties, which gives a decomposition $\begin{array} { r } { \underline { { W } } _ { L } ^ { \mathcal { G } } = \bigsqcup _ { F \not = E } \pi ^ { - 1 } ( \mathbb { P } \check { L } _ { F } ) } \end{array}$ .   
• The virtual Poincar´e polynomial $P _ { \mathbb { P } \mathring { L } _ { F } } ( q )$ of $\mathbb { P } \overset { \circ } { L } _ { F }$ is equal to the reduced characteristic polynomial $\overline { { \chi } } _ { \mathsf { M } / F } ( \boldsymbol { q } ^ { 2 } )$ (see for instance [Kat16, Section 7.2]).

# 4. Formulas for $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$

In this section we will apply our results to the case in which ${ \sf M } = { \sf K } _ { n }$ is the graphic matroid associated to the complete graph on $n$ vertices, and $\mathcal { G } = \mathcal { G } _ { \operatorname* { m i n } }$ . These matroids are often called braid matroids since they admit realizations by the braid arrangements of type $A$ . In this realization, the wonderful variety as recalled in Section 3.5 is the moduli space $\mathcal { M } _ { 0 , n + 1 }$ of $( n + 1 )$ -pointed stable rational curves [DCP95, Section 4.3]. The Poincar´e polynomial $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ of the cohomology ring of $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ therefore satisfies

$$
\mathrm { H } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x ) = P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) .
$$

There have been several works on this Poincar´e polynomial [Kee92, Man95, Get95, AMN24]. We apply our formulas from Section 3 to deduce a new formula for $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ and recover previous results about $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ .

4.1. A formula in terms of partitions. Let us begin with an elementary but useful observation: the flats of $\mathsf { K } _ { n }$ of rank $i$ are in bijection with the partitions of $[ n ] = \{ 1 , \dots , n \}$ into $n - i$ nonempty subsets. The bijection is given by assigning to the partition $S _ { 1 } \sqcup \cdots \sqcup$ $S _ { n - i } = [ n ]$ the set of edges in the union $\mathsf { K } _ { S _ { 1 } } \sqcup \cdots \sqcup \mathsf { K } _ { S _ { n - i } }$ of complete graphs. Thus, the connected flats of $\mathsf { K } _ { n }$ correspond to partitions of $[ n ]$ in which there is at most one part that is not a singleton. The meet (resp. join) on the lattice of flats $\mathcal { L } ( \mathsf { K } _ { n } )$ can be interpreted as the coarsest common refinement (resp. finest common coarsening) of partitions of $[ n ]$ . For braid matroids, because a minor $\mathsf { M } | _ { G } / F$ by flats $F$ and $G$ is again always a direct sum of smaller braid matroids (up to simplification of some parallel elements), one deduces that $\mathcal { G } _ { \mathrm { m i n } } | _ { G } / F$ is the minimal building set for the minor $\mathsf { M } | _ { G } / F$ . From now on, we write $\lambda \vdash [ a ]$ to denote a set partition $\lambda$ of the set $\{ 1 , \ldots , a \}$ .

To apply Theorem 1.3, we need a description of (spanning) $\mathcal { G } _ { \mathrm { m i n } }$ -nested sets of $\mathsf { K } _ { n }$ ; this is accompished by the next proposition. This result has been rediscovered multiple times in the literature; see, e.g., Erd¨os–Sz´ekely [ES89, Theorem 1] or, for a more explicit explanation, [Gai15, Theorem 2.1] by Gaiffi. We include a formulation more adequate for our purposes, and a proof for the sake of completeness.

Proposition 4.1 There is a bijection between the set of spanning $\mathcal { G } _ { \mathrm { m i n } }$ -nested sets of cardinality m in $\mathsf { K } _ { n }$ and the collection of set partitions of $[ n + m - 1 ]$ into m parts, none of which has size one. Moreover, for each spanning $\mathcal { G } _ { \mathrm { m i n } }$ -nested set, the sizes of the parts of the partition are determined as follows: each $F \in S$ corresponds to a part of size $k$ where $\mathsf { K } _ { k }$ is the simplification of the minor $\mathsf { K } _ { n } | _ { F } / \operatorname* { s u p } ( F , S )$ .

Proof. Let us construct an explicit bijection. Consider the set $A = [ n ] \cup \{ w _ { 1 } , \dotsc , w _ { m - 1 } \}$ consisting of $n + m - 1$ elements, i.e., the vertices of $\mathsf { K } _ { n }$ and $m - 1$ additional elements. Consider a spanning $\mathcal { G }$ -nested set $S$ , and linearly order the flats of $S$ as $G _ { 1 } , \ldots , G _ { m }$ first by rank, then, among elements of the same rank, by their minimal element. Reading the list in this order, we build a partition of $A$ as follows:

• If $| S | = 1$ , then the spanning property implies that $S = \{ E \}$ . The corresponding partition of $A$ that we assign in this case is $\{ A \}$ . • If $| S | > 1$ , we set a part in our partition to be $G _ { 1 }$ . To get the remaining parts, proceed as follows. First, replace all the flats $G \in S$ such that $G \supseteq G _ { 1 }$ by $( G \backslash$ $G _ { 1 } ) \cup \{ w _ { 1 } \}$ . Then, construct a partition of size $m - 1$ of $( [ n ] \setminus G _ { 1 } ) \cup \{ w _ { 2 } , \dots , w _ { m - 1 } \}$ : inductively, this corresponds to a $\mathcal { G } _ { \mathrm { m i n } }$ -nested set of cardinality $m - 1$ in $\mathsf { K } _ { n - | G _ { 1 } | }$ .

By ordering the partition of $A$ using the order where all blocks without $w _ { i }$ ’s are less than the blocks with them, and then ordering this second group by the minimal $w _ { i }$ , one can uniquely reconstruct the spanning $\mathcal { G } _ { \mathrm { m i n } }$ -nested set $S$ . 

Example 4.2 Let $n = 6$ and $m = 4$ . To illustrate the bijection in the previous proposition, we list some partitions of length 4 of the set $\{ 1 , 2 , 3 , 4 , 5 , 6 , w _ { 1 } , w _ { 2 } , w _ { 3 } \}$ with at least two elements in each block, along with their corresponding spanning $\mathcal { G } _ { \mathrm { m i n } }$ -nested sets $S$ .

• For $S = \{ 1 2 , 5 6 , 1 2 3 4 , 1 2 3 4 5 6 \}$ , the partition is $\{ \{ 1 , 2 \} , \{ 5 , 6 \} , \{ 3 , 4 , w _ { 1 } \} , \{ w _ { 2 } , w _ { 3 } \} \}$ .   
• For $S = \{ 1 2 , 5 6 , 1 2 5 6 , 1 2 3 4 5 6 \}$ , the partition is $\{ \{ 1 , 2 \} , \{ 5 , 6 \} , \{ w _ { 1 } , w _ { 2 } \} , \{ 3 , 4 , w _ { 3 } \} \}$ .   
• For $S = \{ 1 2 , 5 6 , 3 4 5 6 , 1 2 3 4 5 6 \}$ , the partition is $\{ \{ 1 , 2 \} , \{ 5 , 6 \} , \{ 3 , 4 , w _ { 2 } \} , \{ w _ { 1 } , w _ { 3 } \} \}$ .   
$\bullet$ For $S = \{ 1 2 , 3 4 , 5 6 , 1 2 3 4 5 6 \}$ , the partition is $\{ \{ 1 , 2 \} , \{ 3 , 4 \} , \{ 5 , 6 \} , \{ w _ { 1 } , w _ { 2 } , w _ { 3 } \} \}$ .

We will write $\chi _ { m } ( x ) = ( x - 1 ) ( x - 2 ) \cdot \cdot \cdot ( x - m + 1 )$ for the characteristic polynomial of $\mathsf { K } _ { m }$ , and $\overline { { \chi } } _ { m } ( x ) = ( x - 2 ) \cdot \cdot \cdot ( x - m + 1 )$ for the reduced characteristic polynomial. To avoid overloading the notation, we will also write $\chi _ { \sigma _ { i } } ( x )$ for the polynomial $\chi _ { \left| \sigma _ { i } \right| } ( x )$ whenever $\sigma _ { i }$ denotes a block in a partition of a set.

As an immediate consequence of the bijection described in the previous statement, we obtain the following formula for $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ .

Corollary 4.3 The following identity holds:

$$
P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) = \sum _ { m \geq 1 } \sum _ { \sigma \vdash \left[ n - 1 + m \right] } \prod _ { i = 1 \atop \ell ( \sigma ) = m } ^ { \ell ( \sigma ) } \overline { { \chi } } _ { \sigma _ { i } } ( x ) .
$$

Proof. The assertion follows by combining Theorem 1.3 with the characterization of $\mathcal { G } _ { \mathrm { m i n } }$ nested sets of $\mathsf { K } _ { n }$ proved in Proposition 4.1. 

Now we state a technical lemma that will allow us to rewrite the formula in the previous corollary as a sum over set partitions of $[ n - 1 ]$ instead of $[ n + m - 1 ]$ for varying $m$ .

Lemma 4.4 Let $p _ { \sigma }$ be the number of set partitions of $[ n ]$ of type $\sigma = ( \sigma _ { 1 } , \sigma _ { 2 } , \ldots , \sigma _ { s } ) \vdash n$ . Consider the number $m _ { \sigma }$ of set partitions of $[ n + s ]$ of type $( \sigma _ { 1 } + 1 , \sigma _ { 2 } + 1 , \ldots , \sigma _ { s } + 1 )$ . Then, the following equality holds:

$$
m _ { \sigma } = \frac { ( n + s ) ! } { n ! \prod _ { i = 1 } ^ { s } ( \sigma _ { i } + 1 ) } p _ { \sigma } .
$$

Proof. We must partition $[ n + s ]$ into $s$ parts, none of which has size 1. To do this, first delete $s$ elements from $[ n + s ]$ : there are  n + s  such choices. Call $A$ the set of $n$ numbers in $[ n + s ]$ that remain after the $s$ chosen elements are deleted. We partition $A$ using type $( \sigma _ { 1 } , \dots , \sigma _ { s } )$ in exactly $p _ { \sigma }$ different ways. Each such partition can be augmented to a partition of $[ n + s ]$ of type $( \sigma _ { 1 } + 1 , \ldots , \sigma _ { s } + 1 )$ by adding back the $s$ elements that we removed from $[ n + s ]$ to get $A$ , one to each part. There are $s !$ ways of adding these elements to the parts of $A$ . However, to avoid multiple counting, we must take into account the number of ways that a specific element of a part of our new partition of $[ n + s ]$ could have been picked, i.e., we have to divide by $\textstyle \prod _ { i = 1 } ^ { s } ( \sigma _ { i } + 1 )$ , which leaves us with the equality claimed in the statement. 

Now we put the pieces together to prove Theorem 1.4.

Theorem 1.4 The Poincar´e polynomial of $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ is

$$
P _ { \overline { { \mathcal M } } _ { 0 , n + 1 } } ( x ) = \sum _ { \lambda \vdash [ n - 1 ] } \frac { ( n - 1 + \ell ( \lambda ) ) ! } { ( n - 1 ) ! } \prod _ { i = 1 } ^ { \ell ( \lambda ) } \frac { \overline { { \chi } } _ { \lambda _ { i } + 1 } ( x ) } { \lambda _ { i } + 1 } .
$$

Proof. This follows from Corollary 4.3 using Lemma 4.4.

4.2. A formula by Aluffi–Marcolli–Nascimento. Our next goal is to give a new and self-contained proof of a recent result by Aluffi, Marcolli, and Nascimento [AMN24, Theorem 1.1]. We follow their notation and write $s ( n , k )$ for the (signed) Stirling numbers of the first kind and $S ( \boldsymbol n , \boldsymbol k )$ for the Stirling numbers of the second kind.

Theorem 1.5 The following formula for the Poincar´e polynomial of $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ holds:

$$
P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) = \left( 1 - x \right) ^ { n } \sum _ { k \geq 0 } \sum _ { j \geq 0 } s ( k + n , k + n - j ) S ( k + n - j , k + 1 ) x ^ { k + j } .
$$

Proof. We start from the formula of Corollary 4.3. Since $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ is palindromic of degree $n - 2$ , we obtain

$$
\begin{array} { r l } {  { \boldsymbol { H } _ { \mathcal { H } _ { \mathrm { t o t } + 1 } } ( \boldsymbol { T } ) - \boldsymbol { T } ^ { * * } \boldsymbol { \mathcal { T } } _ { \mathrm { H _ { t o t } } } ( \boldsymbol { T } ^ { * * } ) } } \\ & { = \boldsymbol { T } ^ { * * } \sum _ { \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \ne \boldsymbol { 0 } , \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \ne \boldsymbol { 0 } , \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \ne \boldsymbol { 0 } , \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \ne \boldsymbol { 0 } , \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \mathcal { N } _ { \neq } \mathcal { N } _ { \neq } \dots \mathcal { N } _ { \neq } \mathcal { N } _ { \neq } \dots } } \\ & { = \sum _ { \boldsymbol { \Theta } \ge \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \ne \boldsymbol { 0 } , \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \dots \ne \boldsymbol { 0 } , \boldsymbol { \Theta } \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \dots \times \boldsymbol { \Theta } } \sum _ { \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \dots \times \boldsymbol { \Theta } } ^ { \boldsymbol { \Theta } \cdot 2 } \prod _ { \boldsymbol { \Phi } \ne \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \setminus \boldsymbol { \Theta } } w ^ { - 1 } } \\ &  = ( 1 - w ) ^ { n } \sum _ { \boldsymbol { \Theta } \ge \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \operatorname* { m a x } } \sum _ { \boldsymbol { \Theta } \ne \boldsymbol { 0 } , \atop \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \operatorname* { m a x } } ( \sum _ { \boldsymbol { \Theta } \in \mathcal { N } _ { \neq } \operatorname* { m a x } } ( \boldsymbol { \Theta } - \boldsymbol { \Theta } \boldsymbol { \Theta } \boldsymbol { \Theta } \boldsymbol { \Theta } \boldsymbol { \Theta } \boldsymbol { \Theta }  \end{array}
$$

where the partition $\lambda$ is obtained from $\sigma$ by adding $k$ singletons in all possible ways. By reindexing the first sum, we obtain

$$
P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) = ( 1 - x ) ^ { n } \sum _ { k \geq 0 } \sum _ { \stackrel { \lambda \vdash \left[ n + k \right] } { \ell ( \lambda ) = k + 1 } } x ^ { n - 1 + k } \prod _ { i = 1 } ^ { k + 1 } \chi _ { \lambda _ { i } } ( x ^ { - 1 } ) .
$$

To finish the proof, we note the following elementary identity:

$$
\sum _ { j } x ^ { j } s ( a , a - j ) S ( a - j , b ) = x ^ { a - b } \sum _ { \lambda \vdash [ a ] \atop \ell ( \lambda ) = b } \prod _ { i = 1 } ^ { b } \chi _ { \lambda _ { i } } ( x ^ { - 1 } ) .
$$

To see this, notice that both sides count the number of permutations $\sigma \in \mathfrak { S } _ { a }$ together with a partition of its set of cycles into exactly $b$ blocks. The parameter $- x$ records the number

of cycles of the permutation. Now, applying the identity in Equation (8) to the formula for $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x )$ obtained in Equation (7) yields the equality

$$
P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) = \left( 1 - x \right) ^ { n } \sum _ { k \geq 0 } \sum _ { j \geq 0 } s ( k + n , k + n - j ) S ( k + n - j , k + 1 ) x ^ { k + j } ,
$$

as desired.

4.3. Generating functions and recursions. Several different formulas for the above polynomials (and their exponential generating functions) were obtained by Getzler [Get95], Manin [Man95], and Keel [Kee92]. In this section, we show how our Theorem 3.6, when restricted to ${ \mathsf { M } } = { \mathsf { K } } _ { n }$ and $\mathcal { G } = \mathcal { G } _ { \operatorname* { m i n } }$ , can be used to recover these formulas.

Proposition 4.5 Let $H$ be the exponential generating function of the Poincar´e polynomial of $\overline { { \mathcal { M } } } _ { 0 , n + 1 }$ :

$$
H = \sum _ { n \geq 1 } { P _ { { \overline { { \mathcal M } } } _ { 0 , n + 1 } } ( x ) \frac { t ^ { n } } { n ! } } = \sum _ { n \geq 1 } { \mathrm H } _ { \mathsf K _ { n } } ^ { \mathcal G _ { \operatorname* { m i n } } } ( x ) \frac { t ^ { n } } { n ! } .
$$

(i) The incidence algebra equality $- \overline { { \chi } } ^ { \mathcal { G } _ { \mathrm { m i n } } } \cdot \mathrm { H } ^ { \mathcal { G } _ { \mathrm { m i n } } } = \delta$ is equivalent to the compositional formula [Get95, p. 228]

$$
H \left( t - { \frac { ( 1 + t ) ^ { x } - 1 - x t } { x ( x - 1 ) } } \right) = t .
$$

(ii) The incidence algebra equality $\mathrm { H } ^ { \mathcal { G } _ { \mathrm { m i n } } } \cdot \overline { { \chi } } ^ { \mathcal { G } _ { \mathrm { m i n } } } = - \delta$ is equivalent to the functional equation [Man95, Theorem 0.3.1 (0.7)]

$$
( 1 + H ) ^ { x } = x ^ { 2 } H + 1 - x ( x - 1 ) t .
$$

Proof. The characteristic polynomial is $\overline { { { \chi } } } \kappa _ { n } ( x ) = ( x - 2 ) ( x - 3 ) \cdot \cdot \cdot ( x - n + 1 )$ . Therefore, the exponential generating series of $- \overline { { \chi } } _ { \sf K _ { n } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x )$ can be rewritten in closed form as

$$
G : = \sum _ { n \geq 1 } - \overline { { \chi } } _ { \mathsf { K } _ { n } } ^ { \mathsf { \mathcal { G } } _ { \operatorname* { m i n } } } ( x ) \frac { t ^ { n } } { n ! } = t - \frac { ( 1 + t ) ^ { x } - 1 - x t } { x ( x - 1 ) } .
$$

The compositional formula [Sta99, Theorem 5.1.4] can be applied to the identity $- \overline { { \chi } } ^ { \mathcal { G } _ { \mathrm { m i n } } }$ $\mathrm { H } ^ { \mathcal { G } _ { \mathrm { m i n } } } = \delta$ , from which the statement $H ( G ( t ) ) = t$ follows immediately.

For the second statement, we reason analogously but with the dual identity $\mathrm { H } ^ { \mathcal { G } _ { \mathrm { m i n } } }$ $\overline { { \chi } } ^ { \mathcal { G } _ { \mathrm { m i n } } } = - \delta$ . We obtain that $G ( H ( t ) ) = t$ or, in other words,

$$
H - \frac { ( 1 + H ) ^ { x } - 1 - x H } { x ( x - 1 ) } = t .
$$

The result follows by rearranging the terms.

Proposition 4.6 The polynomial PM $P _ { \overline { { \mathcal { M } } } _ { 0 , n + 1 } } ( x ) = \mathrm { H } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x )$ +1(x) = HGminKn ( is uniquely determined by either of the two following recursions:

$$
\begin{array} { r l r } & { } & { { \displaystyle { \mathrm { H } } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) } = { \mathrm { H } } _ { \mathsf { K } _ { n - 1 } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) + x \sum _ { j = 2 } ^ { n - 1 } \binom { n - 1 } { j } { \mathrm { H } } _ { \mathsf { K } _ { j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) \cdot { \mathrm { H } } _ { \mathsf { K } _ { n - j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) , } \\ & { } & { = ( 1 + x ) { \mathrm { H } } _ { \mathsf { K } _ { n - 1 } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) + \displaystyle { \frac { x } { 2 } } \sum _ { j = 2 } ^ { n - 2 } \binom { n } { j } { \mathrm { H } } _ { \mathsf { K } _ { j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) \cdot { \mathrm { H } } _ { \mathsf { K } _ { n - j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) , } \end{array}
$$

for $n \geq 3$ , and $\mathrm { H } _ { \mathsf { K } _ { 1 } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x ) = \mathrm { H } _ { \mathsf { K } _ { 2 } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x ) = 1$

Proof. The recursion in Equation (9) (and thus, as we will explain below, also Equation (10)) follows from the functional equations proved in the previous statement: this proof is carried out in [Man95, Corollary 0.3.2]. We take the opportunity, however, to prove it combinatorially via the FY-monomials. Consider an FY-monomial with associated $\mathcal { G } _ { \mathrm { m i n } }$ -nested set $S$ , and let $G \in S$ be the smallest flat whose nontrivial part contains $n$ , if it exists. Let k , so that the FY-monomial is be the elements of $S _ { < G }$ and . $H _ { 1 } , \ldots , H _ { h }$ be the elements of $S _ { \ Z G }$ $\Pi _ { i = 1 } ^ { k } x _ { F _ { i } } ^ { e _ { i } } x _ { G } ^ { e } \prod _ { j = 1 } ^ { h } x _ { H _ { j } } ^ { e _ { j } }$ Notice that Qki=1 xeiFixe−1G\{n} is i  jan FY-monomial for the complete induced subgraph on $G \setminus \{ n \}$ and $\Pi _ { j = 1 } ^ { h } x _ { H _ { j } \vee G } ^ { e _ { j } }$ is an FY-monomial for the matroid $\mathsf { K } _ { n } / G$ with respect to the minimal building set. The first summand of Equation (9) corresponds to FY-monomials where such $G$ does not exist; the other summands correspond to the connected flats $G$ of rank $j$ whose nontrivial part contains $n$ .

The recursion appearing in Equation (10) was proved by Keel in [Kee92, p. 550]. It can be deduced from (9) by first rearranging the term in the sum corresponding to $j = n - 1$ :

$$
\begin{array} { r l } & { \displaystyle \mathrm { H } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) = \mathrm { H } _ { \mathsf { K } _ { n - 1 } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) + x \sum _ { j = 2 } ^ { n - 1 } \binom { n - 1 } { j } \mathrm { H } _ { \mathsf { K } _ { j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) \cdot \mathrm { H } _ { \mathsf { K } _ { n - j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) } \\ & { \qquad = ( 1 + x ) \mathrm { H } _ { \mathsf { K } _ { n - 1 } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) + x \sum _ { j = 2 } ^ { n - 2 } \binom { n - 1 } { j } \mathrm { H } _ { \mathsf { K } _ { j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) \cdot \mathrm { H } _ { \mathsf { K } _ { n - j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) . } \end{array}
$$

Then, using the change of variables $i = n - j$ on the sum of the right hand side, this is equivalent to

$$
\begin{array} { r l r } & { } & { \mathrm { H } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) = ( 1 + x ) \mathrm { H } _ { \mathsf { K } _ { n - 1 } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) + x \displaystyle \sum _ { i = 2 } ^ { n - 2 } \binom { n - 1 } { n - i } \mathrm { H } _ { \mathsf { K } _ { n - i } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) \cdot \mathrm { H } _ { \mathsf { K } _ { i } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) } \\ & { } & { = ( 1 + x ) \mathrm { H } _ { \mathsf { K } _ { n - 1 } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) + x \displaystyle \sum _ { j = 2 } ^ { n - 2 } \binom { n - 1 } { j - 1 } \mathrm { H } _ { \mathsf { K } _ { n - j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) \cdot \mathrm { H } _ { \mathsf { K } _ { j } } ^ { \mathcal { G } _ { \operatorname* { m i n } } } ( x ) , } \end{array}
$$

where in the second step we used the identity  n − 1 $\textstyle { \binom { n - 1 } { n - i } } = { \binom { n - 1 } { i - 1 } }$ and the change of variables $i = j$ . Keel’s formula (10) follows from adding Equations (11) and (12), applying Pascal’s identity $\textstyle { \binom { n - 1 } { j } } + { \binom { n - 1 } { j - 1 } } = { \binom { n } { j } }$ , and dividing by two. 

# 5. Conversion between building sets

Given two building sets $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ on $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ , we now explain how to relate $\mathrm { H } _ { \mathsf { M } } ^ { \lessgtr } ( x )$ and $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } ^ { \prime } } ( x )$ . It will suffice to consider the case when $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ differ by one element (see the first part of Proposition 5.2). We thus show the following.

Theorem 5.1 If $\mathcal { G }$ and ${ \mathcal { G } } ^ { \prime } = { \mathcal { G } } \setminus \{ G \}$ are building sets of $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ differing by one element $G$ , then

$$
\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x ) = \mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } ^ { \prime } } ( x ) + ( x + \cdots + x ^ { | f ( \mathcal { G } ^ { \prime } | _ { G } ) | - 1 } ) \mathrm { H } _ { \mathsf { M } | _ { G } } ^ { \mathcal { G } ^ { \prime } | _ { G } } ( x ) \mathrm { H } _ { \mathsf { M } / G } ^ { \mathcal { G } ^ { \prime } / G } ( x ) .
$$

For the proof, we need the following description of $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ as the Chow ring of a fan (see [FY04]). The Bergman fan of $\mathsf { M }$ with respect to $\mathcal { G }$ is the fan $\Sigma _ { \mathcal { G } }$ in $\mathbb { R } ^ { E }$ consisting of the cones ${ \mathbb { R } } _ { \geq 0 } \{ \mathbf { e } _ { G } : G \in S \}$ , one for each $\mathcal { G }$ -nested set $S$ . Here, we denoted by $\mathbf { e } _ { G }$ the sum $\textstyle \sum _ { i \in G } \mathbf { e } _ { i }$ of standard basis vectors. The ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ is equal to the Chow ring $A ^ { \bullet } ( \Sigma _ { \mathcal { G } } )$ of the fan $\Sigma _ { \mathcal { G } }$ . We now note the following.

Proposition 5.2 [FM05, Proof of Theorem 4.2] Let ${ \mathcal { G } } \supset { \mathcal { H } }$ be two building sets of $\mathcal { L } ( \boldsymbol { M } )$ , and let $G$ be a minimal element of $\mathcal { G } \backslash \mathcal { H }$ . Then, ${ \mathcal { G } } ^ { \prime } : = { \mathcal { G } } \setminus \{ G \}$ is a building set. Moreover, the fan $\Sigma _ { \mathcal { G } }$ is the stellar subdivision of the fan $\Sigma _ { \mathcal { G } ^ { \prime } }$ at the face corresponding to the nested set $f ( \mathcal { G } ^ { \prime } | _ { G } )$ of factors of $G$ in $\mathcal { G } ^ { \prime }$ .

We thus need to understand how Chow rings of fans change under stellar subdivisions. The following lemma allows us to do this.

Lemma 5.3 Let $A ^ { \bullet } ( \Sigma )$ denote the Chow ring of a smooth fan $\Sigma$ with rational coefficients. For a cone $\sigma \in \Sigma$ of dimension $\ell$ , let $\widetilde { \Sigma }$ be the stellar subdivision of $\Sigma$ at $\sigma$ , and let $\overline { { \mathrm { s t } } } _ { \sigma } ( \Sigma )$ be the closed star of $\Sigma$ at $\sigma$ (i.e. the fan consisting of the cones of $\Sigma$ containing $\sigma$ ). Then, as graded vector spaces, we have

$$
A ^ { \bullet } ( \widetilde { \Sigma } ) \simeq A ^ { \bullet } ( \Sigma ) \oplus \bigoplus _ { i = 1 } ^ { \ell - 1 } A ^ { \bullet } ( \overline { { { \mathrm { s t } } } } _ { \sigma } ( \Sigma ) ) [ - i ] .
$$

Proof. This follows from [Kee92, Theorem 1 in Appendix], which is stated for Chow rings of blow-ups of smooth varieties along smooth varieties satisfying a certain surjectivity condition (which holds for toric varieties). In our case, the stellar subdivision corresponds to the blow-up of the toric variety $X _ { \Sigma }$ along the torus-invariant subvariety corresponding to the cone $\sigma$ . 

The last remaining ingredient concerns the star fans of Bergman fans.

Lemma 5.4 Let $G$ be a nonempty flat of M not necessarily in a building set $\mathcal { G } ^ { \prime }$ , and let $f ( \mathcal { G } ^ { \prime } | _ { G } ) = \{ G _ { 1 } ^ { \prime } , . . . , G _ { \ell } ^ { \prime } \}$ be the $\mathcal { G } ^ { \prime }$ -factors. Then, the closed star of $\Sigma _ { \mathcal { G } ^ { \prime } }$ at the cone corresponding to $f ( \mathcal { G } ^ { \prime } | _ { G } )$ is isomorphic to the fan $\Sigma _ { \pmb { \mathscr { G } } ^ { \prime } | _ { G _ { 1 } ^ { \prime } } } \times \cdot \cdot \cdot \times \Sigma _ { \pmb { \mathscr { G } } ^ { \prime } | _ { G _ { \ell } ^ { \prime } } } \times \Sigma \pmb { \mathscr { G } } ^ { \prime } / G$ .

Proof. This follows from a repeated application of [BEPV24, Theorem 1.6], which states that for any $G ^ { \prime } \in \mathcal { G } ^ { \prime }$ , the closed star of $\Sigma _ { \mathcal { G } ^ { \prime } }$ at the ray $\mathbb { R } _ { \geq 0 } \{ \mathbf { e } _ { G ^ { \prime } } \}$ is isomorphic to the product $\Sigma _ { \mathcal { G } ^ { \prime } | _ { G ^ { \prime } } } \times \Sigma _ { \mathcal { G } ^ { \prime } / G ^ { \prime } }$ . 

Proof of Theorem 5.1. Let $\{ G _ { 1 } ^ { \prime } , \ldots , G _ { \ell } ^ { \prime } \}$ be the $\mathcal { G } ^ { \prime }$ -factors of $\mathcal { G } ^ { \prime } | _ { G }$ . Note the isomorphism $\Sigma _ { \pmb { \mathscr { G } } ^ { \prime } | _ { G _ { 1 } ^ { \prime } } } \times \cdot \cdot \times \Sigma _ { \pmb { \mathscr { G } } ^ { \prime } | _ { G _ { \ell } ^ { \prime } } } \times \Sigma _ { \pmb { \mathscr { G } } ^ { \prime } / G } \simeq \Sigma _ { \pmb { \mathscr { G } } ^ { \prime } | _ { G } } \times \Sigma _ { \pmb { \mathscr { G } } ^ { \prime } / G }$ . Combining the second part of Proposition 5.2 with the two preceding lemmas yields the desired result.

# 6. The failure of inequalities

In this section, we discuss the potential failure of the properties in Figure 1 for the Hilbert series of the ring $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ .

6.1. Failure of log-concavity. We provide examples where the coefficients of $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ do not form a log-concave sequence. It will be useful to consider $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ as the Poincar´e polynomial of a wonderful variety constructed via a sequential blow-up, which was reviewed in Section 3.5. The strategy common to all of our examples is the following simple observation: if a sequence $( a , b , c )$ is log-concave but $2 b < a + c$ (i.e. not concave), then $( a + \ell , b + \ell , c + \ell )$ is not log-concave for all large $\ell > 0$ , since $( b + \ell ) ^ { 2 } - ( a + \ell ) ( c + \ell ) = ( b ^ { 2 } - a c ) + ( 2 b - a - c ) \ell$ .

Example 6.1 ( $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ need not have log-concave coefficients) Let $\mathsf { M }$ be the uniform matroid of rank 9 on the ground set $E = [ 8 ] \sqcup [ { \overline { { 1 0 } } } ]$ of 18 elements, where $[ 8 ] = \{ 1 , \dots , 8 \}$ and $[ \overline { { 1 0 } } ] = \{ \overline { { 1 } } , \dots , \overline { { 1 0 } } \}$ . Let

$$
\mathcal { G } = E \cup \{ 1 2 , 1 2 3 , 1 2 3 4 , \dots , 1 2 3 4 5 6 7 8 \} \cup \binom { [ \overline { { 1 0 } } ] } { 8 } \cup \{ E \} ,
$$

which one verifies to be a building set on $\mathsf { M }$ by noting that every interval $[ \sigma , F ]$ for a proper flat $F \neq E$ is a Boolean lattice.

As uniform matroids are realizable over $\mathbb { C }$ , following the construction of wonderful varieties given in Section 3.5, we find that the Chow polynomial $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ is the Poincar´e polynomial of the variety obtained as a sequential blow-up of $\mathbb { P } ^ { \mathrm { 8 } }$ , in two steps:

(1) First, one blows up a complete flag $\mathbb { P } L _ { 1 2 3 4 5 6 7 8 } \subsetneq \cdots \subsetneq \mathbb { P } L _ { 1 2 } \subsetneq \mathbb { P } L _ { 1 }$ in $\mathbb { P } ^ { 8 }$ , starting with the point, then (the strict transform of) the line, then the plane, etc. (2) Then, one blows up ${ \binom { 1 0 } { 8 } } = 4 5$ more points not lying on the hyperplane $\mathbb { P } L _ { 1 }$ .

By induction, one verifies that blowing up a complete flag in a projective space yields a variety whose Betti numbers are the binomial coefficients. In our case, we have that after step (1) the Betti numbers read $( 1 , 8 , 2 8 , 5 6 , 7 0 , 5 6 , 2 8 , 8 , 1 )$ . Note that $2 \cdot 2 8 - ( 5 6 + 8 ) =$ $- 8 < 0$ . Then, for step (2), when one blows up a point, we note that the Betti numbers change by adding 1 to each of the entries that are not the first or the last. In our case, one may also deduce this geometric fact from Theorem 5.1. In particular, the final Betti numbers read $( 1 , 8 + 4 5 , 2 8 + 4 5 , 5 6 + 4 5 , . . . )$ ), and at the third entry, we find that logconcavity fails since $7 3 ^ { 2 } - 5 3 \cdot 1 0 1 = - 2 4 < 0$ .

Example 6.2 $\mathrm { \cdot } \mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x )$ need not have log-concave coefficients) The geometry in the previous example of blowing up a complete flag in $\mathbb { P } ^ { \mathrm { 8 } }$ and then blowing up 45 additional general points can be realized as a minimal building set on a matroid in the following way. We do this in two steps, corresponding to the steps (1) and (2) in the geometry described in the previous example. We will use notions in matroid theory such as free extensions and principal extensions by flats, which can be found in [Oxl11, Chapter 7].

(1) First, we construct a matroid $\mathsf { M } _ { 9 }$ of rank 9 on the ground set $\{ 1 , 2 , 2 ^ { \prime } , 3 , 3 ^ { \prime } , . . . , 9 , 9 ^ { \prime } \}$ as follows. Let $\mathsf { M } _ { 1 } = \mathsf { U } _ { 1 , \{ 1 \} }$ , the Boolean matroid of rank 1 on the set $\{ 1 \}$ . Then, for $i > 1$ , we inductively define $\mathsf { M } _ { i } : = ( \mathsf { M } _ { i - 1 } \oplus i ) + i ^ { \prime }$ . That is, $\mathsf { M } _ { i }$ is obtained from $\mathsf { M } _ { i - 1 }$ by adding a coloop named $i$ , and then taking the free extension of $\mathsf { M } _ { i - 1 } \oplus i$ by an additional element $i ^ { \prime }$ . In general, for a loopless matroid $\mathsf { M }$ on ground set $E$ , the connected flats of the free extension $\mathsf { M } + e$ are the proper connected flats of $\mathsf { M }$ and $E \cup e$ . Hence, we find that the connected flats of $\mathsf { M } _ { 9 }$ are $\{ 1 , 1 2 2 ^ { \prime } , 1 2 2 ^ { \prime } 3 3 ^ { \prime } , \ldots , 1 2 2 ^ { \prime } 3 3 ^ { \prime } \ldots 9 9 ^ { \prime } \}$ . That is, the matroid $\mathsf { M } _ { 9 }$ has the property that its connected flats form a complete chain in $\mathcal { L } ( \boldsymbol { \mathsf { M } } _ { 9 } )$ .

(2) Now, we extend the matroid $\mathsf { M } _ { 9 }$ (on a ground set of size 19) to a matroid (of the same rank, 9) on $1 9 + 4 5 \cdot 9 = 4 2 4$ elements, such that the set of connected flats will consist of those from $\mathsf { M } _ { 9 }$ together with an additional 45 flats of corank 1. For a matroid $\mathsf { M }$ of rank $r$ on a ground set $E$ , denote by $\mathsf { M } + \mathsf { U } _ { r - 1 , r }$ the matroid on ground set $E \sqcup [ r ]$ constructed as follows. First, perform $( r - 1 )$ -many times the free extension of $\mathsf { M }$ to obtain $\mathsf { M } + \{ 1 , \ldots , r - 1 \}$ , and then take the principal extension of the resulting matroid by the flat $\{ 1 , \ldots , r - 1 \}$ . (That is, we extend M by $r$ -many elements as freely as possible with the sole constraint that the $r$ -many elements that were added do not form a basis). Note that if a flat $F \subseteq E \sqcup [ r ]$ of $\mathsf { M } + \mathsf { U } _ { r - 1 , r }$ properly intersects $[ r ]$ , then $F \cap [ r ]$ consists of coloops of $F$ , since by construction no element of $[ r ]$ is in the closure of any proper flat of $\mathsf { M }$ . Thus, the proper connected flats of $\mathsf { M } + \mathsf { U } _ { r - 1 , r }$ are the proper connected flats of M and $[ r ]$ . The matroid we seek is $\mathsf { M } _ { 9 } + \mathsf { U } _ { 8 , 9 } + \ldots + \mathsf { U } _ { 8 , 9 }$ (where $+ \mathsf { U } _ { 8 , 9 }$ is performed 45 times).

This matroid on 424 elements can be realized over any infinite field as follows. Begin with a $9 \times 1 9$ matrix of the form

$$
A _ { 9 } = \left[ \begin{array} { c c c c c c c } { 1 } & { 0 } & { * } & { 0 } & { * } & { 0 } & { * } & \\ & { 1 } & { 1 } & { 0 } & { * } & { 0 } & { * } & \\ & & & { 1 } & { 1 } & { 0 } & { * } & \\ & & & & { 1 } & { 1 } & & \\ & & & { \vdots } & & & { \ddots } \end{array} \right] ,
$$

where the $^ *$ entries are generic. (The row span of) this matrix realizes the matroid $\mathsf { M } _ { 9 }$ . Then, for $i = 1 , \dots , 4 5$ , let $U _ { i }$ each be a distinct generic choice of a $9 \times 9$ matrix of rank 8. The matroid we seek is realized by (the row span of) the matrix

$$
[ A _ { 9 } | U _ { 1 } | \cdots | U _ { 4 5 } ] .
$$

The resulting wonderful variety corresponding to the minimal building set is the sequential blow-up of $\mathbb { P } ^ { \aleph }$ identical to the one described in Example 6.2, and hence the Chow polynomial $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ fails to have log-concave coefficients.

Example 6.3 $( \mathrm { H } _ { \mathsf { U } _ { n , n } } ^ { \mathcal { G } } ( x )$ need not have log-concave coefficients) We provide a building set on a Boolean matroid for which the Chow polynomial has non-log-concave coefficients. That is, the $h$ -vector of the corresponding nestohedron need not be log-concave. For this connection to nestohedra, we point to [PRW08].

First, let us consider the following geometric scenario. For $d > 0$ and $N > d$ , we consider blowing up a flag of subspaces in $\mathbb P ^ { N }$ of dimensions $0 , 1 , \ldots , d$ . Note that for any $n > 0$ , the first $n$ Betti numbers of this blow-up are independent of $N$ provided that $N$ is large enough. In fact, they are the partial sums of binomial coefficients:

$$
\left( { \binom { d + 1 } { 0 } } , { \binom { d + 1 } { 0 } } + { \binom { d + 1 } { 1 } } , { \binom { d + 1 } { 0 } } + { \binom { d + 1 } { 1 } } + { \binom { d + 1 } { 2 } } + { \binom { d + 1 } { 2 } } , \ldots \right) .
$$

For instance, when $d \ : = \ : 5$ , the sequence reads $( 1 , 7 , 2 2 , 4 2 , \dots )$ . Then, if we blow-up additional points, for instance 39 additional points, we obtain that $( 2 2 + 3 9 ) ^ { 2 } - ( 7 + 3 9 ) \cdot$ · $( 4 2 + 3 9 ) = - 5 < 0$ .

This geometric scenario can be realized as the construction of a wonderful variety from a building set on a Boolean matroid, as follows. Take $\mathsf { M } = \mathsf { U } _ { 4 5 , [ 4 5 ] }$ , the Boolean matroid of rank 45 on the set [45]. The collection of subsets $\{ [ 4 4 ] , [ 4 3 ] , \dots , [ 3 9 ] \} \cup \{ [ 4 5 ] \setminus \{ i \} : i \in$ [39]} along with the atoms and the ground set [45] forms a building set. The ordering ([44], [43], . . . , [39], [45] $\backslash \{ 1 \} , \ldots , [ 4 5 ] \backslash \{ 3 9 \} )$ refines the partial order by reverse inclusion, and the resulting sequential blow-up gives the desired geometric scenario above.

6.2. Koszulity, gamma-positivity, and real-rootedness. A result by Coron [Cor25b, Theorem 3.12] states that if the lattice of flats $\mathcal { L } ( \boldsymbol { \mathsf { M } } )$ of a matroid is supersolvable, then the Chow ring $D ( \mathsf { M } , \mathcal { G } _ { \operatorname* { m i n } } )$ is Koszul. This generalizes a result by Dotsenko [Dot22] for the braid matroid $\mathsf { K } _ { n }$ . On the other hand, Mastroeni and McCullough showed that for all matroids $\mathsf { M }$ , the ring $D ( \mathsf { M } , \mathcal { G } _ { \mathrm { m a x } } )$ is Koszul.

As is explained in [FMSV24, Section 5.4], whenever the degree of the Hilbert series of a graded Artinian Koszul algebra is not more than 4, then it has only real zeros. Furthermore, by [FMSV24, Theorem 1.8], for any matroid M we have that $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } _ { \mathrm { m a x } } } ( x )$ is $\gamma$ -positive.

The following question arises naturally in this context.

Question 6.4 Is there a loopless matroid M and a building set $\mathcal { G }$ such that $D ( \mathsf { M } , \mathsf { \mathcal { G } } )$ is Koszul and $\mathrm { H } _ { \mathsf { M } } ^ { \mathcal { G } } ( x )$ fails to be

(i) $\gamma$ -positive, or (ii) real-rooted?

We note that the real-rootedness question for $\mathrm { H } _ { \mathsf { K } _ { n } } ^ { \mathcal { G } _ { \mathrm { m i n } } } ( x )$ was posed by Aluffi, Chen, and Marcolli in [ACM24, Conjecture 1], while in this special case, $\gamma$ -positivity follows directly from Proposition 4.6 (see [ACM24, Theorem 1.2]).

# References

[ACM24] Paolo Aluffi, Stephanie Chen, and Matilde Marcolli, Log concavity of the Grothendieck class of $\overline { { \mathcal { M } } } _ { 0 , n }$ , arXiv e-prints (2024), arXiv:2402.02646.   
[ADH23] Federico Ardila, Graham Denham, and June Huh, Lagrangian geometry of matroids, J. Amer. Math. Soc. 36 (2023), no. 3, 727–794.   
[AHK18] Karim Adiprasito, June Huh, and Eric Katz, Hodge theory for combinatorial geometries, Ann. of Math. (2) 188 (2018), no. 2, 381–452.   
[AMN24] Paolo Aluffi, Matilde Marcolli, and Eduardo Nascimento, Explicit formulas for the Grothendieck class of ${ \overline { { \mathcal { M } } } } _ { 0 , n }$ , arXiv e-prints (2024), arXiv:2406.13095.   
[ANR25] Robert Angarone, Anastasia Nathanson, and Victor Reiner, Chow rings of matroids as permutation representations, J. Lond. Math. Soc. (2) 111 (2025), no. 1, Paper No. e70039.   
[BEPV24] Sarah Brauner, Christopher Eur, Elizabeth Pratt, and Raluca Vlad, Wondertopes, arXiv eprints (2024), arXiv:2403.04610.   
[BES24] Spencer Backman, Christopher Eur, and Connor Simpson, Simplicial generation of Chow rings of matroids, J. Eur. Math. Soc. (JEMS) 26 (2024), no. 11, 4491–4535.   
[BHM $^ +$ 22] Tom Braden, June Huh, Jacob P. Matherne, Nicholas Proudfoot, and Botong Wang, A semismall decomposition of the Chow ring of a matroid, Adv. Math. 409 (2022), Paper No. 108646.   
[Br¨a15] Petter Br¨and´en, Unimodality, log-concavity, real-rootedness and beyond, Handbook of enumerative combinatorics, Discrete Math. Appl. (Boca Raton), CRC Press, Boca Raton, FL, 2015, pp. 437– 483.   
[Bre94] Francesco Brenti, Log-concave and unimodal sequences in algebra, combinatorics, and geometry: an update, Jerusalem combinatorics ’93, Contemp. Math., vol. 178, Amer. Math. Soc., Providence, RI, 1994, pp. 71–89.   
[BV25] Petter Br¨and´en and Lorenzo Vecchi, Chow polynomials of uniform matroids are real-rooted, arXiv e-prints (2025), arXiv:2501.07364.   
[CHL $^ +$ 25] Colin Crowley, June Huh, Matt Larson, Connor Simpson, and Botong Wang, The Bergman fan of a polymatroid, Trans. Amer. Math. Soc. (2025), to appear.   
[Cor25a] Basile Coron, Matroids, Feynman categories, and Koszul duality, Duke Math. J. (2025), to appear.   
[Cor25b] , Supersolvability of built lattices and Koszulness of generalized Chow rings, Compos. Math. (2025), to appear.   
[DCP95] Corrado De Concini and Claudio Procesi, Wonderful models of subspace arrangements, Selecta Math. (N.S.) 1 (1995), no. 3, 459–494.   
[Dot22] Vladimir Dotsenko, Homotopy invariants for ${ \overline { { \mathcal { M } } } } _ { 0 , n }$ via Koszul duality, Invent. Math. 228 (2022), no. 1, 77–106.   
[ER95] Richard Ehrenborg and Margaret A. Readdy, Sheffer posets and $r$ -signed permutations, Ann. Sci. Math. Qu´ebec 19 (1995), no. 2, 173–196.   
[ES89] P´eter L. Erd¨os and L. A. Sz´ekely, Applications of antilexicographic order. I. An enumerative theory of trees, Adv. in Appl. Math. 10 (1989), no. 4, 488–496.   
[FM05] Eva Maria Feichtner and Irene M¨uller, On the topology of nested set complexes, Proc. Amer. Math. Soc. 133 (2005), no. 4, 999–1006.   
[FMSV24] Luis Ferroni, Jacob P. Matherne, Matthew Stevens, and Lorenzo Vecchi, Hilbert-Poincar´e series of matroid Chow rings and intersection cohomology, Adv. Math. 449 (2024), Paper No. 109733.   
[FMV24] Luis Ferroni, Jacob P. Matherne, and Lorenzo Vecchi, Chow functions for partially ordered sets, arXiv e-prints (2024), arXiv:2411.04070.   
[FS24] Luis Ferroni and Benjamin Schr¨oter, Valuative invariants for large classes of matroids, J. Lond. Math. Soc. (2) 110 (2024), no. 3, Paper No. e12984, 86.   
[Ful93] William Fulton, Introduction to toric varieties, Annals of Mathematics Studies, vol. 131, Princeton University Press, Princeton, NJ, 1993, The William H. Roever Lectures in Geometry.   
[FY04] Eva Maria Feichtner and Sergey Yuzvinsky, Chow rings of toric varieties defined by atomic lattices, Invent. Math. 155 (2004), no. 3, 515–536.   
[Gai15] Giovanni Gaiffi, Nested sets, set partitions and Kirkman-Cayley dissection numbers, European J. Combin. 43 (2015), 279–288.   
[Get95] E. Getzler, Operads and moduli spaces of genus 0 Riemann surfaces, The moduli space of curves (Texel Island, 1994), Progr. Math., vol. 129, Birkh¨auser Boston, Boston, MA, 1995, pp. 199–230.   
[Her72] A. P. Heron, Matroid polynomials, Combinatorics (Proc. Conf. Combinatorial Math., Math. Inst., Oxford, 1972), Inst. Math. Appl., Southend-on-Sea, 1972, pp. 164–202.   
[Hos24] Elena Hoster, The Chow and augmented Chow polynomials of uniform matroids, arXiv e-prints (2024), arXiv:2410.22329.   
[HRS21] Thomas Hameister, Sujit Rao, and Connor Simpson, Chow rings of vector space matroids, J. Comb. 12 (2021), no. 1, 55–83.   
[Kat16] Eric Katz, Matroid theory for algebraic geometers, Nonarchimedean and tropical geometry, Simons Symp., Springer, [Cham], 2016, pp. 435–517.   
[Kee92] Sean Keel, Intersection theory of moduli space of stable n-pointed curves of genus zero, Trans. Amer. Math. Soc. 330 (1992), no. 2, 545–574.   
[Lia24] Hsin-Chieh Liao, Equivariant $\gamma$ -positivity of Chow rings and augmented Chow rings of matroids, arXiv e-prints (2024), arXiv:2408.00745.   
[Man95] Yu. I. Manin, Generating functions in algebraic geometry and sums over trees, The moduli space of curves (Texel Island, 1994), Progr. Math., vol. 129, Birkh¨auser Boston, Boston, MA, 1995, pp. 401–417.   
[Oxl11] James Oxley, Matroid theory, second ed., Oxford Graduate Texts in Mathematics, vol. 21, Oxford University Press, Oxford, 2011.   
[PP23] Roberto Pagaria and Gian Marco Pezzoli, Hodge theory for polymatroids, Int. Math. Res. Not. IMRN (2023), no. 23, 20118–20168.   
[PRW08] Alex Postnikov, Victor Reiner, and Lauren Williams, Faces of generalized permutohedra, Doc. Math. 13 (2008), 207–273.   
[Rot71] Gian-Carlo Rota, Combinatorial theory, old and new, Actes du Congr\`es International des Math´ematiciens (Nice, 1970), Tome 3, 1971, pp. 229–233.   
[Sch03] Alexander Schrijver, Combinatorial optimization. Polyhedra and efficiency. Vol. B, Algorithms and Combinatorics, vol. 24,B, Springer-Verlag, Berlin, 2003, Matroids, trees, stable sets, Chapters 39–69.   
[Sta89] Richard P. Stanley, Log-concave and unimodal sequences in algebra, combinatorics, and geometry, Graph theory and its applications: East and West (Jinan, 1986), Ann. New York Acad. Sci., vol. 576, New York Acad. Sci., New York, 1989, pp. 500–535.   
[Sta99] Enumerative combinatorics. Vol. 2, Cambridge Studies in Advanced Mathematics, vol. 62, Cambridge University Press, Cambridge, 1999, With a foreword by Gian-Carlo Rota and appendix 1 by Sergey Fomin.   
[Sta12] Enumerative combinatorics. Volume 1, second ed., Cambridge Studies in Advanced Mathematics, vol. 49, Cambridge University Press, Cambridge, 2012.   
[Stu24] Christian Stump, Chow and augmented Chow polynomials as evaluations of Poincar´e-extended ab-indices, arXiv e-prints (2024), arXiv:2406.18932.   
[Wel76] Dominic J. A. Welsh, Matroid theory, L. M. S. Monographs, No. 8, Academic Press [Harcourt Brace Jovanovich, Publishers], London-New York, 1976.

(C. Eur) Carnegie Mellon University, Pittsburgh, PA, USA Email address: ceur@cmu.edu

(L. Ferroni) Institute for Advanced Study, Princeton, NJ, USA, and Universita di Pisa, \` Pisa, Italy Email address: ferroni@ias.edu

(J. P. Matherne) North Carolina State University, Raleigh, NC, USA Email address: jpmather@ncsu.edu

(R. Pagaria) Universita di Bologna, Bologna, Italy \` Email address: roberto.pagaria@unibo.it

(L. Vecchi) KTH Royal Institute of Technology, Stockholm, Sweden Email address: lvecchi@kth.se